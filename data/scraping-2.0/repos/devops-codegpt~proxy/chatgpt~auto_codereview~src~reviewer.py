#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 10:24
# @Author  : Jack
# @File    : reviewer.py
# @Software: PyCharm

import os
from typing import Iterable


from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

# Fill your openai api key.
os.environ["OPENAI_API_KEY"] = ""


def chatgpt_review(filename: str, content: str) -> Iterable:
    template = """Below is the code content of the file "{filename}", The number before each line of code is the code line number mark:
    {content}
    Please assume you are a professional reviewer of this code. tell me the problems with the above code and your suggestions for this. You can only answer with this format: file:line:type:details. file is code file name,  line is The number of the line for which the problem is, The value of type is Warn or Error, depending on the severity of the problem, The value of details is your suggestion or problem description.
    """
    prompt_template = PromptTemplate(template=template, input_variables=["filename", "content"])
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    resp = chain.run({
        "filename": filename,
        "content": content
    })
    comments = str(resp).split("\n")
    return comments
