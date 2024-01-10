import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import ConversationChain

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# Conversation
# ---------------------------
llm = ChatOpenAI(temperature=0)

# ------
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there, my name is Sharon!")
print(output)

output = conversation.predict(
    input="What would be a good company name for a company that makes colorful socks?"
)
print(output)

output = conversation.predict(input="What is my name?")
print(output)

output = conversation.predict(input="Who are you in this conversation?")
print(output)
