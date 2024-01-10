#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################
# -------- 2、文本模板 ----------------
# 创建步骤：
#  1、确定模板和用户输入参数的位置，利用PromptTemplate类创建一个文本模板；
#  2、给聊天模板传入参数，生成完整的信息
#########################################

from langchain import PromptTemplate

# ---------------{{一个没有输入参数的模板案例}}-----------------------------------------------------------------------------------------
no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
result = no_input_prompt.format()
print(result)  # -> "Tell me a joke."


# ---------------{{有一个输入参数的模板案例}}-------------------------------------------------------------------------------------------
one_input_prompt = PromptTemplate(input_variables=["adjective"], template="Tell me a {adjective} joke.")
result = one_input_prompt.format(adjective="funny")
print(result)  # -> "Tell me a funny joke."


# ---------------{{有多个输入参数的模板案例}}--------------------------------------------------------------------------------------------
multiple_input_prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template="Tell me a {adjective} joke about {content}."
)
result = multiple_input_prompt.format(adjective="funny", content="chickens")
print(result) # -> "Tell me a funny joke about chickens."


# ---------------{{如果您不想通过input_variables}}-------------------------------------------------------------------------------------
template = "Tell me a {adjective} joke about {content}."
prompt_template = PromptTemplate.from_template(template)

prompt_str = prompt_template.input_variables
print("模板参数：{}".format(prompt_str)) # -> ['adjective', 'content']

result = prompt_template.format(adjective="funny", content="chickens")
print(result) # -> Tell me a funny joke about chickens.