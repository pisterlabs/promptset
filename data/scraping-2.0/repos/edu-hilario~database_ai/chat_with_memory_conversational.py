from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

from utils.load_openai_api_key import load_openai_api_key

load_openai_api_key()

template = """
You are a chatbot designe to be a helpful
Your goal is to help the user achieve understanding of a topic
Take what the user is saying and cooperate

[chat_history]:
{chat_history}
Human: {human_input}
Chatbot:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=ChatOpenAI(), prompt=prompt, verbose=True, memory=memory)


def chatbot(question):
    llm_chain.predict(human_input=question)


while True:
    question = input("Enter your question or 'exit' to quit: ")
    if question.lower() == "exit":
        print("Exiting...")
        break
    else:
        try:
            print(chatbot(question))
        except Exception as e:
            print(f"Damn, ran into a snag: {e}")
