import configparser
import pprint

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
import json

config = configparser.ConfigParser()
config.read('../config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

llm = ChatOpenAI(openai_api_key=openai_api_key)
output_parser = StrOutputParser()


def simple_chain():
    global prompt, llm, output_parser, chain
    prompt = ChatPromptTemplate.from_template(
        "帮我写一首关于{topic}的诗"
    )
    chain = prompt | llm | output_parser
    # response = chain.invoke(input={"topic": "大海"})
    # print(response)

    # response = chain.batch([{"topic": "老鼠"}, {"topic": "狗"}])
    # pp = pprint.PrettyPrinter
    # pp.pprint(response)

    for t in chain.stream({"topic": "bears"}):
        print(t)


simple_chain()

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema.runnable import RunnableMap


def complex_chain():
    global retriever, prompt, chain
    vectorDB = DocArrayInMemorySearch.from_texts(
        [
            "人是由恐龙进化而来",
            "熊猫喜欢吃天鹅肉"
        ],
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)
    )
    retriever = vectorDB.as_retriever()
    retriever.get_relevant_documents("人从哪里来？")
    template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    inputs = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    })
    chain = inputs | prompt | llm | output_parser
    response = chain.invoke(input={"question": "人从哪里来"})
    print(response)

# complex_chain()
