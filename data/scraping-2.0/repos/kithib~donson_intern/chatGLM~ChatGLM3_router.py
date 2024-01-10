import json
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional
import conf

class ChatGLM(LLM):
    tokenizer: object = None
    model: object = None
    max_length = conf.MAX_LENGTH
    top_p = conf.TOP_P
    temperature = conf.TEMPERATURE
    def __init__(self):
        super().__init__()

    def set_args(self, max_length=None, top_p=None, temperature=None):
        if max_length: self.max_length = max_length
        if top_p: self.top_p = top_p
        if temperature: self.temperature = temperature

    def handle_prompt(self, prompt):
        query = prompt[-1]['content']
        history = prompt[:-1]
        # for i in range(0, len(prompt) - 1, 2):
        #   history.append((prompt[i]['content'], prompt[i + 1]['content']))
        return query, history

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"

    def load_model(self, model_name_or_path = conf.CHATGLM_MODEL_PATH):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True, device_map="auto",
        ).half()


    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        return ""

    def chat(self, prompt):
            if isinstance(prompt, list):
            #logger.info("本次回答使用的是 ChatGLM 模型，使用的 prompt 为：{}".format(prompt))
            # 记录程序开始时间
            #start_time = time.time()
                query, history = self.handle_prompt(prompt)
                response, _ = self.model.chat(
                    self.tokenizer,
                    query,
                    history=history,
                    max_length=self.max_length,
                    top_p=self.top_p,
                    temperature=self.temperature
                    )
                #end_time = time.time()
                #elapsed_time = end_time - start_time
                #logger.info("推理耗时:{}".format(elapsed_time))
                return response
            else:
                return "Wrong Type"

    def stream_chat(self, prompt):
        if isinstance(prompt, list):
        #logger.info("本次回答使用的是 ChatGLM 模型，使用的 prompt 为：{}".format(prompt))
            query, history = self.handle_prompt(prompt)
        for response, history in self.model.stream_chat(self.tokenizer, query,
                                                      history=history):
            yield response
        else:
            return "Wrong Type"
