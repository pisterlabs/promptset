from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

from utils.openapi import count_tokens

load_dotenv()

questions = [
    "Good morning AI!",
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge",
    "I just want to analyze the different possibilities. What can you think of?",
    "Which data source types could be used to give context to the model?",
    "What is my aim again?",
]

if __name__ == "__main__":
    llm = OpenAI(model_name="text-davinci-003", temperature=0)
    chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())

    for question in questions:
        count_tokens(chain, question)

    print(chain.memory.buffer)
