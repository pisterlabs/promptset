from langchain import PromptTemplate, LLMChain
from langchain.llms import AzureOpenAI, OpenAIChat

# from config import *
#
# # llm = NewAzureOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
# llm = AzureOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)
# #
# # # llm.deployment_name = 'ChatGPT-0301'
# llm.deployment_name = 'code-davinci-002'
# from langchain import PromptTemplate
#
# template = """Answer the question based on the context below. If the
# question cannot be answered using the information provided answer
# with "I don't know".
#
# Context: Large Language Models (LLMs) are the latest models used in NLP.
# Their superior performance over smaller models has made them incredibly
# useful for developers building NLP enabled applications. These models
# can be accessed via Hugging Face's `transformers` library, via OpenAI
# using the `openai` library, and via Cohere using the `cohere` library.
#
# Question: {query}
#
# Answer: """
#
# prompt_template = PromptTemplate(
#     input_variables=["query"],
#     template=template
# )
#
# print(prompt_template.format(query="Which libraries and model providers offer LLMs?"))
# from langchain.llms import AzureOpenAI
# llm = AzureOpenAI(deployment_name="text-davinci-002")
# llm.deployment_name='code-davinci-002'
# from langchain.chains import LLMChain
# chain = LLMChain(llm=llm, prompt=prompt_template)
#
# # Run the chain only specifying the input variable.
# print(chain.run("Which libraries and model providers offer LLMs?"))



#定义系统角色
# prefix_messages = [{"role": "system", "content": "你是一位乐于助人的历史学教授，名叫王老师"}]
#
# # 定义大型语言模型llm
# llm = OpenAIChat(model_name='gpt-3.5-turbo',
#                  temperature=0,
#                  prefix_messages=prefix_messages,
#                  max_tokens = 256)
# template = """根据问题: {user_input}
# 请用生动有趣，简明扼要的方式回答上述问题。"""
#
# prompt = PromptTemplate(template=template,input_variables=["user_input"])
#
# #定义chain
# llm_chain = LLMChain(prompt=prompt, llm=llm)
#
# user_input = "中国一共有哪些朝代？"
#
# print(llm_chain.run(user_input))

#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://adt-openai.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY","938ce9d50df942d08399ad736863d063")
from config import *
# response = openai.ChatCompletion.create(
#     engine="ChatGPT-0301",
#     messages = [
#         {
#             "role": "system",
#             "content": "You are an AI assistant that helps people find information."
#         },
#         {
#             "role": "user",
#             "content": "根据上下文回答问题:\n今天是2015年2月21日星期三\n问题:\n"
#         },
#         {
#             "role": "user",
#             "content": "今天星期几?"
#         }
#     ],
#     temperature=0.7,
#     max_tokens=800,
#     top_p=0.95,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=None)
#
# print(response)


openai.api_type = "azure"
openai.api_base = "https://adt-openai.openai.azure.com/"
openai.api_version = "2022-12-01"
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY", '938ce9d50df942d08399ad736863d063')


prompt='''
<|im_start|>system
You are an AI assistant that helps people find information.
<|im_end|>
<|im_start|>user
请阅读以下内容，然后结合你的知识，用代码方式返回。
内容:Working with Highlight annotations


FRDocReloadPage(frDocument, 0, FALSE);

问题:如何创建annot
<|im_end|>
<|im_start|>assistant
'''

response = openai.Completion.create(
    engine="code-davinci-002",
    prompt=prompt,
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["<|im_end|>"])
print("问题: {}".format('xxx'))
print("回答: {}".format(response["choices"][0]["text"]))
