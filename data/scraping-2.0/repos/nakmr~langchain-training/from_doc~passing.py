from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

vectorstore = Chroma.from_texts(
    ["harrison worked at kensho", "bears like to eat honey", "honey is sweet"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

template = """
    Answer the question based only on the following context: {context}
    Quetion: {question}
    """
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = retrieval_chain.invoke("What do bears like to eat?")
print(result)