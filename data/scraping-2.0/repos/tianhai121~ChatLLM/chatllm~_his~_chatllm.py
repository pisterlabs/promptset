#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : chatllm
# @Time         : 2023/4/20 15:46
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :
import types
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

# ME
from meutils.pipe import *


class ChatLLM(LLM):
    """
    from llm.utils import llm_load

    model, tokenizer = llm_load("/Users/betterme/PycharmProjects/AI/CHAT_MODEL/chatglm")
    glm = ChatLLM()
    glm.chat_func = model.chat # partial(self.chat_func, **kwargs)
    """
    chat_func: Callable = None
    history = []
    max_turns = 3

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    # todo: https://github.com/hwchase17/langchain/issues/2415 增加流失

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        result = self.chat_func(query=prompt, history=self.history[-self.max_turns:])
        response = history = None
        if isinstance(result, types.GeneratorType):
            for response, history in tqdm(result, desc='Stream'):
                yield response, history
        else:
            response, history = result

        if stop:
            response = enforce_stop_tokens(response, stop)
        self.history = history  # 历史所有         self.history += [[None, response]]

        return response

    def set_chat_kwargs(self, **kwargs):
        self.chat_func = partial(self.chat_func, **kwargs)


if __name__ == '__main__':
    from chatllm.utils import llm_load

    model, tokenizer = llm_load("/CHAT_MODEL/chatglm")
    glm = ChatLLM()
    glm.chat_func = partial(model.chat, tokenizer=tokenizer)
    glm.chat_func = partial(model.stream_chat, tokenizer=tokenizer)
