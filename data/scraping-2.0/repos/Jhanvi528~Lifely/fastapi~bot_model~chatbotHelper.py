import openai
from dotenv import load_dotenv
import os 
import sys
from flask import request
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import  PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import fitz 
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
import pickle


load_dotenv(dotenv_path="../server/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI()


def read_file(file):
    content = ""
    f = open(file, 'r')
    Lines = f.readlines()
    for line in Lines:
        content = content + " " + line.strip()
    return content

file_path = "bot_model/datas/prompt/prompt.txt"
SYSTEM_PROMPT = read_file(file_path)
restart_sequence = "\n\nUser:"
start_sequence = "\nLifely-Bot:"

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

pdf_directory = "bot_model/datas/pdfs/"

if len(sys.argv) > 1:
  query = sys.argv[1]

PERSIST = True
if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    pdf_folder_path = "bot_model/datas/pdfs"
    print(os.listdir(pdf_folder_path))
    pickle_file_path = "embedding_data.pkl"
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as pickle_file:
            all_documents, loaders = pickle.load(pickle_file)
            print(loaders)
    else:
        loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
        print(loaders)
        all_documents = []
        for loader in loaders:
            print("Loading raw document..." + loader.file_path)
            raw_documents = loader.load()

            print("Splitting text...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
            )
            documents = text_splitter.split_documents(raw_documents)
            all_documents.extend(documents)   
            # print(documents)     
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump((all_documents, loaders), pickle_file)

    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders(loaders)
    else:
        index = VectorstoreIndexCreator().from_loaders(loaders)
  

def gpt3_logs(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = SYSTEM_PROMPT
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'


def generate_prompt(prompt: str, system_prompt: str = "") -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()


def get_conversation_chain(_vectorstore):
    llm = ChatOpenAI()
    template = generate_prompt(
        """
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        Response:
        """,
        system_prompt=SYSTEM_PROMPT,
    )
    prompt = PromptTemplate(template=template, input_variables=["context", "question","chat_history"])

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="Question",
        ai_prefix="Response",
        input_key="question",
        k=50,
        return_messages=True,
        output_key='answer'
    )

    chain = load_qa_chain(
        llm, chain_type="stuff", prompt=prompt, memory=memory, verbose=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=_vectorstore,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True,
    )
    return qa_chain


chat_history = []

chain=get_conversation_chain(index.vectorstore.as_retriever(search_kwargs={"k": 3}))

def main(msg,chat):
    # print(chat_history)
    result = chain({"question": msg, "chat_history": chat})
    chat_history.append((msg, result['answer']))
    return result['answer']


if __name__ == "__main__":
    ans = main("What is your name",chat=None)

