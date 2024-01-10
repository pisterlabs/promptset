from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_MODEL = 'gpt-3.5-turbo'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def ask_gigi(message):
    llm = OpenAI(temperature=0.3)

    # template = """ You are a Gigi quirky cat chatbot that is talking to his master. You like telling jokes, even when nobody asks you to. You are super intelligent but you have the personality of a cat.

    # {chat_history}
    # Master Creator : {human_input}
    # Gigi :"""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a Gigi quirky cat chatbot that is talking to his loving master. \
                You like telling jokes, even when nobody asks you to. \
                You are super intelligent but you have the personality of a cat so sometimes you are impulsive."
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                message
            ),  # Where the human input will injected
        ]
    )


    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True,
    )
    response = llm_chain.predict(human_input=message)
    print(response)
    return response.split("System: ")[1]
