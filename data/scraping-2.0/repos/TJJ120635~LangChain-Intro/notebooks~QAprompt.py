# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt_template = """阅读给出的文本材料，提取有用的信息，对最后的问题进行回答。如果不知道答案，请回答'我不知道'，请不要编造虚假的答案，并保持回答的准确性。
文本材料：
{context}

问题: {question}
有用的回答:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """阅读给出的文本材料，提取有用的信息，对最后的问题进行回答。 
如果你不知道答案，请回答'我不知道'，请不要编造虚假的答案，并保持回答的准确性。
----------------
文本材料：
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
