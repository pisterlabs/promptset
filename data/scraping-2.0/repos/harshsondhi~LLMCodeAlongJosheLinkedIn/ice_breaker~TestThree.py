import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.cache import InMemoryCache
from langchain import PromptTemplate
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
api_key = "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"
llm = OpenAI()
chat = ChatOpenAI(openai_api_key=api_key)

# result = chat([SystemMessage(content='You are a rude teenager who only wants to party and not answer questions'),
#     HumanMessage(content='Tell me a fact about pluto')])
# print(result.content)


# result = chat.generate([
#     [SystemMessage(content='You are a rude teenager who only wants to party and not answer questions'),
#     HumanMessage(content='Tell me a fact about pluto')],
#     [SystemMessage(content='You are a friendly assistant'),
#     HumanMessage(content='Tell me a fact about pluto')]
#
# ])
#
# print(result.generations[0][0].text)
langchain.llm_cache = InMemoryCache()

# result = chat([SystemMessage(content='You are a friendly assistant'),
#     HumanMessage(content='Tell me a fact about pluto')],temperature=.05,max_tokens=40)
# print(result.content)

# print(llm.predict('Tell me a fact about Mars'))
#
# print(llm.predict('Tell me a fact about Mars'))
# multi_input_prompt = PromptTemplate(template= 'Tell me fact about {topic} for a {level} student', input_variables=["topic","level"])
# response = llm(multi_input_prompt.format(topic='the ocean', level='10rd grade'))
# print(response)

system_template = (
    "You are vet doctor that specializes in {animal}  15 weeks old and can give health tip for next  {"
    "future_time}"
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "when should we expect  periods  for female kitten"
human_mesage_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_mesage_prompt]
)
prompt = chat_prompt.format_prompt(
    animal="kitten", future_time="6 months"
).to_messages()
result = chat(prompt)
print(result.content)
