import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Vectorstore
vectorstore = Chroma("langchain_store", OpenAIEmbeddings(), persist_directory="./data/CHROMA_DB_2")

# Load GPT-3.5-turbo-1106
llm = ChatOpenAI()

# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a chatbot that is used to talk to users about a simulation platform called Fogify. Fogify is an emulation Framework easing the modeling, deployment and experimentation of fog testbeds. Fogify provides a toolset to: model complex fog topologies comprised of heterogeneous resources, network capabilities and QoS criteria; deploy the modelled configuration and services using popular containerized infrastructure-as-code descriptions to a cloud or local environment; experiment, measure and evaluate the deployment by injecting faults and adapting the configuration at runtime to test different what-if scenarios that reveal the limitations of a service before introduced to the public." ),
        MessagesPlaceholder(variable_name="chat_history"),
        SystemMessagePromptTemplate.from_template(
            "Context: {context}"
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Memory
memory = ConversationBufferMemory(memory_key='chat_history',input_key="question", max_length=3000, return_messages=True)

# LLM Chain
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False, memory=memory)

def remove_extra_spaces(text):
    return " ".join(text.split())

def get_context(question):
    #Get the most similar documents
    docs = vectorstore.similarity_search(question)

    context = ''
    for doc in docs:
        doc.page_content = remove_extra_spaces(doc.page_content)
        if(len(context) + len(doc.page_content) > 6000):
            break
        context += ' '+doc.page_content
    return context

def get_question():
    question = input("\nUser: ")
    return question

def ask_ai(question):
    context = get_context(question)
    dict = {"question": question, "context": context}
    response = llm_chain(dict)
    print("AI: " + response.get("text"))

def main():
    while True:
        question = get_question()
        if(question == 'exit'):
            break
        ask_ai(question)

if __name__ == "__main__":
    main()

