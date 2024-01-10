from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

vectorstore = DocArrayInMemorySearch.from_texts(
    [
        "harrison worked at kensho",
        "bears like to eat honey"
    ],
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

response = chain.invoke("where did harrison work?")
print(response)
