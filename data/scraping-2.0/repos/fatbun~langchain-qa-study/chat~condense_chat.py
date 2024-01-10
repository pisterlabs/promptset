#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Project : langchang-openai
# @Time : 2023/8/19 15:39
# @Author : Ben Li.
# @File: condense_chat.py
from langchain import LLMChain
from langchain.llms import ChatGLM
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


def condense_question(input, history) -> str:
    chain = LLMChain(
        llm=ChatGLM(
            endpoint_url="http://127.0.0.1:8000/chat",
            temperature=0.5),
        prompt=CONDENSE_QUESTION_PROMPT
    )
    return chain.run(question=input, chat_history=history)


if __name__ == '__main__':
    answer = condense_question("怎么开单", [["什么是超级车店", "超级车店是一个汽车管理系统，包括接车开单、会员管理、预约管理"]])
    print(type(answer))
    print(answer)
