from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

template = "你是一位专业顾问，负责为专注于{product}的公司起名。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "公司主打产品是{product_detail}。"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

prompt = prompt_template.format_prompt(product='鲜花装饰', product_detail='创新的鲜花设计。').to_messages()

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()
result = chat(prompt)
print(result)
