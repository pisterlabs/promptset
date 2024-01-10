import os
import dotenv
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import AzureOpenAIEmbeddings
from chromadb.config import Settings
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

dotenv.load_dotenv()

# Load the text of the Geneva Convention.
loader = PyPDFLoader("https://ihl-databases.icrc.org/assets/treaties/365-GC-I-EN.pdf")
pages = loader.load_and_split()



deployment_name = os.getenv("EMBEDDING_MODEL_NAME")
embeddings = AzureOpenAIEmbeddings(azure_deployment=deployment_name)
vectorstore = Chroma.from_documents(
    documents=pages,
    embedding=embeddings,
    client_settings=Settings()
)
retriever = vectorstore.as_retriever()


temperature = 0.2
deployment_name = os.getenv("MODEL_NAME")
llm = AzureChatOpenAI(temperature=temperature, azure_deployment=deployment_name)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_template = PromptTemplate.from_template("""
Context:
  {context}
Question: {question}""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)


def produce_response(question, history):
    """
      question: user input, normally it is a question asked
      history: chat history
    """
    return rag_chain.invoke(question)


demo = gr.ChatInterface(
  produce_response,
  title="OpenAI Chatbot Example",
  description="A chatbot example for QCRI Generative AI Hackathon 2023",
  )

if __name__ == "__main__":
    demo.launch()
