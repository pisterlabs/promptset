from abc import ABC

from langchain.llms.base import LLM
import random
import torch
import torch.nn as nn
import transformers
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from typing import Optional, List, Dict, Any
from models.loader import LoaderCheckPoint
from models.extensions.callback import (Iteratorize, Stream, FixedLengthQueue)
import models.shared as shared
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)


def _streaming_response_template() -> Dict[str, Any]:
    """
    :return: 响应结构
    """
    return {
        "text": ""
    }


def _update_response(response: Dict[str, Any], stream_response: str) -> None:
    """Update response from the stream response."""
    response["text"] += stream_response


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class LLamaLLM(BaseAnswer, LLM, ABC):
    checkPoint: LoaderCheckPoint = None
    history = []
    history_len: int = 3
    max_new_tokens: int = 500
    num_beams: int = 1
    temperature: float = 0.5
    top_p: float = 0.4
    top_k: int = 10
    repetition_penalty: float = 1.2
    encoder_repetition_penalty: int = 1
    min_length: int = 0
    logits_processor: LogitsProcessorList = None
    stopping_criteria: Optional[StoppingCriteriaList] = None

    state: object = {'max_new_tokens': 50,
                     'seed': 1,
                     'temperature': 0, 'top_p': 0.1,
                     'top_k': 40, 'typical_p': 1,
                     'repetition_penalty': 1.2,
                     'encoder_repetition_penalty': 1,
                     'no_repeat_ngram_size': 0,
                     'min_length': 0,
                     'penalty_alpha': 0,
                     'num_beams': 1,
                     'length_penalty': 1,
                     'early_stopping': False, 'add_bos_token': True, 'ban_eos_token': False,
                     'truncation_length': 2048, 'custom_stopping_strings': '',
                     'cpu_memory': 0, 'auto_devices': False, 'disk': False, 'cpu': False, 'bf16': False,
                     'load_in_8bit': False, 'wbits': 'None', 'groupsize': 'None', 'model_type': 'None',
                     'pre_layer': 0, 'gpu_memory_0': 0}

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "LLamaLLM"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    def encode(self, prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
        input_ids = self.checkPoint.tokenizer.encode(str(prompt), return_tensors='pt',
                                                     add_special_tokens=add_special_tokens)
        # This is a hack for making replies more creative.
        if not add_bos_token and input_ids[0][0] == self.checkPoint.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]

        # Llama adds this extra token when the first character is '\n', and this
        # compromises the stopping criteria, so we just remove it
        if type(self.checkPoint.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]

        # Handling truncation
        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        return input_ids.cuda()

    def decode(self, output_ids):
        reply = self.checkPoint.tokenizer.decode(output_ids, skip_special_tokens=True)
        return reply

    def generate_with_callback(self, callback=None, **kwargs):
        self.checkPoint.clear_torch_cache()
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        with torch.no_grad():
            self.checkPoint.model.generate(**kwargs)
            print("方法结束")

    def generate_with_streaming(self, **kwargs):
        return Iteratorize(self.generate_with_callback, kwargs)

    # 将历史对话数组转换为文本格式
    def history_to_text(self):
        formatted_history = ''
        history = self.history[-self.history_len:] if self.history_len > 0 else []
        for i, (old_query, response) in enumerate(history):
            formatted_history += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        return formatted_history

    def generate_softprompt_history_tensors(self):
        """
        历史对话软提示
            这段代码首先定义了一个名为 history_to_text 的函数，用于将 self.history
            数组转换为所需的文本格式。然后，我们将格式化后的历史文本
            再用 self.encode 将其转换为向量表示。最后，将历史对话向量与当前输入的对话向量拼接在一起。
        :return:
        """

        # 对话内容
        # 处理历史对话
        formatted_history = self.history_to_text()
        return formatted_history

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"__call:{prompt}")
        #  向量拼接
        softprompt = self.generate_softprompt_history_tensors()
        history = self.history[-self.history_len:] if self.history_len > 0 else []
        softprompt += "[Round {}]\n问：{}\n答：".format(len(history), prompt)

        softprompt_input_ids = self.checkPoint.tokenizer.encode(str(softprompt), return_tensors='pt')

        input_ids = self.encode(prompt, add_bos_token=self.state['add_bos_token'], truncation_length=self.max_new_tokens)

        # 全连接层,在矩阵相乘运算中，要求第一个矩阵的列数与第二个矩阵的行数相等
        llm_proj = nn.Linear(
            softprompt_input_ids.shape[1], self.checkPoint.model_config.hidden_size
        )
        # 确保数据类型匹配
        softprompt_input_ids = softprompt_input_ids.to(torch.float32)
        # 隐藏层输入
        inputs_llm = llm_proj(softprompt_input_ids).to(self.checkPoint.llm_device)
        # 隐藏层形状
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(self.checkPoint.llm_device)

        # 设置填充符、截止符、开始符号、
        self.checkPoint.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.checkPoint.tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.checkPoint.tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.checkPoint.tokenizer.add_special_tokens({'unk_token': '</s>'})
        llm_tokens = self.checkPoint.tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.checkPoint.llm_device)

        # 输入
        inputs_embeds = self.checkPoint.model.get_input_embeddings()(llm_tokens.input_ids)
        # 输入张量形状对齐
        inputs_llm_broadcasted = inputs_llm.unsqueeze(1).expand_as(inputs_embeds)
        inputs_embeds = torch.cat([inputs_llm_broadcasted, inputs_embeds], dim=1)
        # 输入掩码张量形状对齐
        atts_llm_broadcasted = atts_llm.unsqueeze(1).expand_as(llm_tokens.attention_mask)
        attention_mask = torch.cat([atts_llm_broadcasted, llm_tokens.attention_mask], dim=1)

        if self.logits_processor is None:
            self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(InvalidScoreLogitsProcessor())
        # Find the eos tokens
        eos_token_ids = [self.checkPoint.tokenizer.eos_token_id] if self.checkPoint.tokenizer.eos_token_id is not None else []
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "encoder_repetition_penalty": self.encoder_repetition_penalty,
            "min_length": self.min_length,
            "temperature": self.temperature,
            "eos_token_id": eos_token_ids,
            "logits_processor": self.logits_processor}

        # 对话模型prompt
        gen_kwargs.update({'inputs_embeds': llm_tokens.input_ids})
        # 注意力掩码
        gen_kwargs.update({'attention_mask': llm_tokens.attention_mask})
        if self.stopping_criteria is None:
            self.stopping_criteria = transformers.StoppingCriteriaList()
        # 观测输出
        gen_kwargs.update({'stopping_criteria': self.stopping_criteria})
        shared.stop_everything = False
        stopped = False
        response_template = _streaming_response_template()

        # TODO 此流输出方法需要重写！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # stopping_criteria方法不可控制 迭代器的变量无法共享
        with self.generate_with_streaming(**gen_kwargs) as generator:
            last_reply_len = 0
            reply_index = 0
            # Create a FixedLengthQueue with the desired stop sequence and a maximum length.
            queue = FixedLengthQueue(stop)
            for output in generator:
                new_tokens = len(output) - len(input_ids[0])
                reply = self.decode(output[-new_tokens:])

                new_reply = len(reply) - last_reply_len
                output_reply = reply[-new_reply:]
                queue.add(reply_index, output_reply)
                queue.contains_replace_sequence()
                if stop:
                    pos = queue.contains_stop_sequence()
                    if pos != -1:
                        shared.stop_everything = True
                        stopped = True

                #print(f"{reply_index}：reply  {output_reply}")
                english_reply = queue.put_replace_out(reply_index)
                #print(f"{reply_index}：english_reply  {english_reply}")
                _update_response(response_template, english_reply)
                last_reply_len = len(reply)

                reply_index += 1
                if new_tokens == self.max_new_tokens - 1 or stopped:
                    break

                if output[-1] in eos_token_ids:
                    break

        response = response_template['text']
        print(f"response:{response}")
        self.history = self.history + [[None, response]]
        return response

    def _generate_answer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False,
                         generate_with_callback: AnswerResultStream = None) -> None:
        if history:
            self.history = history
        # Create the StoppingCriteriaList with the stopping strings
        self.stopping_criteria = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        self.stopping_criteria.append(listenerQueue)
        # TODO 需要实现chat对话模块和注意力模型，目前_call为langchain的LLM拓展的api，默认为无提示词模式，如果需要操作注意力模型，可以参考chat_glm的实现

        response = self._call(prompt=prompt, stop=['\n###'])
        answer_result = AnswerResult()
        answer_result.history = self.history
        if listenerQueue.listenerQueue.__len__() > 0:
            answer_result.listenerToken = listenerQueue.listenerQueue.pop()
        answer_result.llm_output = {"answer": response}
        generate_with_callback(answer_result)
