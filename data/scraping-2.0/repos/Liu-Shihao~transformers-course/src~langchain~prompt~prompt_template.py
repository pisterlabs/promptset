from langchain.prompts import PromptTemplate, ChatPromptTemplate

"""
https://python.langchain.com/en/latest/modules/prompts/getting_started.html
to_string： 可以打印传递给LLM的原始内容
to_messages： 传递给 ChatModel 时调用的（它需要一个消息列表）
"""
string_prompt = PromptTemplate.from_template("tell me a joke about {subject}")

chat_prompt = ChatPromptTemplate.from_template("tell me a joke about {subject}")

string_prompt_value = string_prompt.format_prompt(subject="soccer")
print(string_prompt_value)
# text='tell me a joke about soccer'

# to_string()函数可以打印原始内容
print(string_prompt_value.to_string())
# tell me a joke about soccer

chat_prompt_value = chat_prompt.format_prompt(subject="soccer")
print(chat_prompt_value)
# messages=[HumanMessage(content='tell me a joke about soccer', additional_kwargs={}, example=False)]

print(chat_prompt_value.to_string())
# Human: tell me a joke about soccer

print(string_prompt_value.to_messages())
# [HumanMessage(content='tell me a joke about soccer', additional_kwargs={}, example=False)]

print(chat_prompt_value.to_messages())
# [HumanMessage(content='tell me a joke about soccer', additional_kwargs={}, example=False)]
