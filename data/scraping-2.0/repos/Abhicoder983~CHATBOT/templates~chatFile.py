from langchain.llms import OpenAI
import os
def openaiKey(key):
    os.environ["OPENAI_API_KEY"] = key
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
url = []
text = ''
vectoreStore= ''
vectoreStoreURL = None
vectoreStorePDF = None
"""owner_data = "Owner - Mayank"
embedding = OpenAIEmbeddings()
vectoreStore = FAISS.from_documents(owner_data, embedding)"""

'''
agent = OpenAI(temperature = 1)
conv = RetrievalQAWithSourcesChain(llm=agent,retriever = vectoreStore.as_retriever(), memory=ConversationSummaryMemory())
'''
def web(urls):
    global vectoreStoreURL, conv, agent, url
    try:
        url = []
        url.append(urls)
        print("url - 1")
        loaders = UnstructuredURLLoader(urls=url)
        print("url - 2")
        data = loaders.load()
        print("url - 3")

        splitter = CharacterTextSplitter(separator='\n',
                                         chunk_size=500,
                                         chunk_overlap=20)
        print("url - 4")
        docs = splitter.split_documents(data)
        print("url - 5")
        embedding = OpenAIEmbeddings()
        print("url - 6")
        vectoreStoreURL = FAISS.from_documents(docs, embedding)
        print("url - 7")
    except:
        pass

def read_pdf(pdfs):
    global vectoreStorePDF
    try:
        text = ''
        for pdf in pdfs:
            pdf_reader = PdfReader(pdf)
            for pages in pdf_reader.pages:
                text += pages.extract_text()
        splitter = CharacterTextSplitter(separator='\n',
                                         chunk_size=500,
                                         chunk_overlap=20)
        print("url - 4")
        docs = splitter.split_documents(text)
        print("url - 5")
        embedding = OpenAIEmbeddings()
        print("url - 6")
        vectoreStorePDF = FAISS.from_documents(docs, embedding)
        print("url - 7")
    except:
        pass
def chat_connection(temperature = 1):
    print("1")
    global vectoreStoreURL, vectoreStorePDF, conv, agent
    print("2")
    if vectoreStoreURL != None and vectoreStorePDF != None:
        vectoreStore = vectoreStoreURL+vectoreStorePDF
    if vectoreStoreURL != None:
        vectoreStore = vectoreStoreURL
    if vectoreStorePDF != None:
        vectoreStore = vectoreStorePDF
    print("3")
    agent = OpenAI(temperature=temperature)
    print("4")
    conv = RetrievalQAWithSourcesChain.from_llm(llm=agent, retriever=vectoreStore.as_retriever())
    print("5")


def chatnew(chat):
    global conv
    response = conv({"question": chat}, return_only_outputs=True)
    answer = response['answer']
    return answer
"""
web("https://en.wikipedia.org/wiki/French_Revolution")
chat_connection()
print(chatnew("hi"))
"""