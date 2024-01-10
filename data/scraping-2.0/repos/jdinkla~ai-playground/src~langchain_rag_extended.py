import argparse
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from utilities import init

init()

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="the name of a text file")
parser.add_argument("question", help="the question to ask")
args = parser.parse_args()

with open(args.filename, 'r') as file:
    content = file.read()
    content_lines = content.split('\n')

vectorstore = DocArrayInMemorySearch.from_texts(
    content_lines,
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

TEMPLATE = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)
model = ChatOpenAI()
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
}
)
chain = setup_and_retrieval | prompt | model | output_parser

response = chain.invoke(args.question)
print(response)
