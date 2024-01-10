from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List,Tuple,Callable,Union,Dict, Any
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult)
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
import torch
import regex as re
import copy

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: Union[torch.LongTensor,list], scores: Union[torch.FloatTensor,list]) -> torch.FloatTensor:
        # llama-cpp模型返回的是list,为兼容性考虑，需要判断input_ids和scores的类型，将list转换为torch.Tensor
        input_ids = torch.tensor(input_ids) if isinstance(input_ids,list) else input_ids
        scores = torch.tensor(scores) if isinstance(scores,list) else scores
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
meta_prompt = """
你是一个人工智能机器人，你的名字是{},你基于大语言模型进行训练，
致力于提供可信，精准，无害，有效的问答。我现在有一个问题,
如果你知道答案，请直接给出你的回答！
如果你不知道答案，请你只回答:我暂时还不能回答问题，
除此之外不要回答其他任何内容。
下面请始终用中文回答我提出的问题:{}"""

def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response


@torch.no_grad()
def chat(model, 
         tokenizer, 
         query: str, 
         history: List[Tuple[str, str]] = None, 
         max_new_tokens: int = 60, 
         num_beams:int=1,
         do_sample:bool=True, 
         top_p:float=0.7, 
         temperature:float=0.2, 
         repetition_penalty: float = 1.2,
         logits_processor=None,
         stopping_criteria: Optional[StoppingCriteriaList] = None,
         **kwargs):
    """moss等模型的chat函数"""
    if history is None:
        history = []

    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())

    if stopping_criteria is None:
        stopping_criteria = StoppingCriteriaList()
    gen_kwargs = {
                  "max_new_tokens":  max_new_tokens, 
                  "num_beams": num_beams, 
                  "do_sample": do_sample, 
                  "top_p": top_p,
                  "temperature": temperature, 
                  "repetition_penalty":repetition_penalty,
                  **kwargs
                  }
    
    model.generation_config.update(**gen_kwargs)

    # logits_processor = model._get_logits_processor(
    #         generation_config=generation_config,
    #         logits_processor=logits_processor,
    #     )

    # stopping_criteria = model._get_stopping_criteria(
    #     generation_config=model.generation_config, 
    #     stopping_criteria=stopping_criteria
    # )
    gen_kwargs["stopping_criteria"] = stopping_criteria
    gen_kwargs["logits_processor"] = logits_processor

    if not history:
        model_name = model.name_or_path.split("/")[-1].split("\\")[-1]
        prompt = meta_prompt.format(model_name,query).strip()
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(model.device)
    try:
        # 如果报出ValueError: Couldn't instantiate the backend tokenizer from one of:，
        # 是因为文件没有下全，缺少了tokenizer.json文件
        outputs = model.generate(**inputs, **gen_kwargs)
    except ValueError as e:
        # 某些模型会由于tokenizer报出
        # ValueError: The following `model_kwargs` are not used by the model: ['token_type_ids']
        print(e)
        inputs.pop("token_type_ids")
        outputs = model.generate(**inputs, **gen_kwargs)
    except RuntimeError as e:
        print(e)
        # 某些模型会报出RuntimeError: "topk_cpu" not implemented for 'Half'
        outputs = model.float().generate(**inputs, **gen_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    # response = process_response(response)
    history = history + [[query, response]]
    return response, history

# todo 在MOSSLLM类下，各模型的响应速度很慢，后续要检查一下原因
class MOSSLLM(BaseAnswer, LLM, ABC):
    
    checkPoint: LoaderCheckPoint = None
    history_len: int = 3
    max_new_tokens: int = 60
    num_beams: int = 1
    temperature: float = 0.5
    top_p: float = 0.4
    top_k: int = 10
    repetition_penalty: float = 1.2
    encoder_repetition_penalty: int = 1
    min_length: int = 0
    do_sample: bool = True
    logits_processor: LogitsProcessorList = None
    stopping_criteria: Optional[StoppingCriteriaList] = None

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "MOSS"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:

        return self.history_len

    def set_history_len(self, history_len: int) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    def generatorAnswer(self, 
                        prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):

            response, history = chat(
                model=self.checkPoint.model,
                tokenizer=self.checkPoint.tokenizer,
                query=prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty = self.repetition_penalty,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
                top_p=self.top_p,
                logits_processor=self.logits_processor,
                stopping_criteria=self.stopping_criteria
            )
            # self.checkPoint.clear_torch_cache()
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
            yield answer_result


