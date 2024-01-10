#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################
# -------- 2、聊天模板 ----------------
# 和文本模板的区别是，每条消息都带有role信息。
# 创建步骤：
#  1、确定模板和用户输入参数的位置，转换为字符串 or 利用PromptTemplate类创建一个文本模板；
#  2、根据1的结果，利用MessagePromptTemplate相关的类 创建出带有role信息的模板
#  3、通过 ChatPromptTemplate.from_messages 将多个MessagePromptTemplate的实例构建成一个完整的聊天模板
#  4、给聊天模板传入参数，生成llm需要的输入
#########################################

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# ---------------{{根据字符串生成模板}}------------------------------------------------------------------------------------
template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text},end!!"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# ---------------{{先生成'文本模板'，在生成聊天模板}}-------------------------------------------------------------------------
prompt=PromptTemplate(
    template="You are a helpful assistant that translates {input_language} to {output_language}.",
    input_variables=["input_language", "output_language"],
)
system_message_prompt_2 = SystemMessagePromptTemplate(prompt=prompt)

assert system_message_prompt == system_message_prompt_2

# ---------------{{使用模板}}---------------------------------------------------------------------------------------------
# 创建聊天模板，输入多个模块，会按给定顺序格式化
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt,system_message_prompt_2])

# 从格式化消息中获取聊天完整信息
promptValue = chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.")
# 转换为字符串输出
print(promptValue.to_messages())