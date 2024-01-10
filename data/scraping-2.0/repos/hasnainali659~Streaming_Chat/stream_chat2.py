# from langchain.llms import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.4)

def chat_response(query):
    # Prompt
    prompt = ChatPromptTemplate.from_template("You are ICS Arabia chatbot so answer {query} accordingly")

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # # Chain
    # chatbot = LLMChain(
    #     llm=llm,
    #     prompt=prompt,
    #     memory=memory
    # )

    runnable = prompt | llm | StrOutputParser()

    # query = "What is AI in 1000 words"
    for chunk in runnable.stream({"query": query}):
        # print(chunk, end="", flush=True)
        yield chunk.content

chat_response("Write an essay of 1000 words on AI")