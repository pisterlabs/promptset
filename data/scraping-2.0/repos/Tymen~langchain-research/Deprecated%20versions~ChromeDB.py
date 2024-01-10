import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
import getpass
import os

os.environ['OPENAI_API_KEY'] = "sk"

# import dotenv

# dotenv.load_dotenv()

loader = WebBaseLoader(
    web_paths=("https://siip.app/FAQ.html", "https://siip.group/privacyverklaring/")
    # bs_kwargs=dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("post-content", "post-title", "post-header")
    #     )
    # ),
)
docs = loader.load()



file_path='./instruction.json'
data = json.loads(Path(file_path).read_text())
pprint(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke("Worden persoons gegevens verwijderd na een bepaalde tijd? en hoe kom ik in contact met siip voor vragen ?"))

# cleanup
vectorstore.delete_collection()