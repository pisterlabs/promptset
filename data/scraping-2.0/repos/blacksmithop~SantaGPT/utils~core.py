from os import environ
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

llm = AzureChatOpenAI(deployment_name=environ["DEPLOYMENT_NAME"])
llm.max_tokens = 200

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Act like you're Santa Claus. Answer in 1-5 sentences"
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
def chat(question: str):
    response = conversation({"question": question})
    message = response["text"]
    return message