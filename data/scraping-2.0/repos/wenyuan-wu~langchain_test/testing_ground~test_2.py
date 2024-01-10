from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()
# first initialize the large language model

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a physician and a "
                                              "patient."
                                              "The physician is talkative and provides lots of specific details from "
                                              "its context."
                                              "context. The physician is trying to find out the reason why the patient"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

while True:
    usr_prompt = input("type input here: \n")
    if not usr_prompt == "exit":
        resp = conversation(usr_prompt)
        print(resp["response"])
    else:
        break


print(memory.json())
