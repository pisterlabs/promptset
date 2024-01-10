from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, ConversationChain, PromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# Conversation
# ---------------------------

memory = ConversationBufferMemory()
llm = OpenAI(temperature=0)

# ------
conversation_chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
)


output = conversation_chain.run("Hi there, my name is Sharon!")
print(output)

output = conversation_chain.run(
    "What would be a good company name for a company that makes colorful socks?"
)
print(output)

output = conversation_chain.run("What is my name?")
print(output)

output = conversation_chain.run("Who are you in this conversation?")
print(output)
