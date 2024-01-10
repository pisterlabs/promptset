import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Chat in a Chain with Prompt Template
# OPEN AI API - POST https://api.openai.com/v1/chat/completions
# ---------------------------

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "The following is a friendly conversation between a human and an AI (named Terence). The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)
print(chat_prompt)

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

output = chain.predict(input="Hi there, my name is Sharon!")
print(output)

output = chain.predict(
    input="What would be a good company name for a company that makes colorful socks?"
)
print(output)

output = chain.predict(input="What is my name?")
print(output)

output = chain.predict(input="Who are you in this conversation? And what is your name?")
print(output)
