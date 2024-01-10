import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from typing import Union
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import time
import subprocess
import qrcode
from pdf2image import convert_from_bytes
import timm
images1 = convert_from_bytes(open(
    '/home/csgrad/sunilruf/nlp_cse/LLM_bot/data/grad-handbook-2023.pdf', 'rb').read())

documents = []
for file in os.listdir("docs3"):
    if file.endswith('.txt'):
        text_path = "docs3/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        pdf_path = "data/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")
vectordb = FAISS.from_documents(documents, embeddings)
try:
    print("Entered context handbook updation")
    with open('context_handbook.txt', 'r') as file:
        data = file.read().replace('\n', '')
except:
    print("No data in context_handbook")
try:
    vectordb.add_texts([str(data)])
except:
    print("Context handbook added to vectordb")
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(max_tokens=150,temperature=0.1, model_name="gpt-3.5-turbo-16k"),
    vectordb.as_retriever(search_type = "similarity_score_threshold", search_kwargs={'score_threshold': 0.5, 'k': 4}),
    return_source_documents=True,
    verbose=False
)
def LLMResponse(query):
    
    print(query)
    chat_history=[]
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"
    while True:
        #query = input(f"{green}Prompt: ")
        #query = "What are the requirements for PhD students?"
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
            break
        if query == '':
            continue
        
        if "correct yourself" in query.lower() or "update" in query.lower() or "modify" in query.lower():
            result = {}
            date = datetime.today().strftime('%Y-%m-%d')
            query = query.replace('correct yourself', '')
            vectordb.add_texts(["Updated info as of "+str(date)+" :"+query])
            chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(max_tokens=150,temperature=0.1, model_name="gpt-3.5-turbo-16k"),
                                                          vectordb.as_retriever(search_kwargs={"k": 4}),verbose=False)
            result['answer'] = "The information is updated.Thank you"
            
            with open("context_handbook.txt", "a") as context_file:
                context_file.write("Updated info as of " + str(date) + ": " + query + "\n")
            
            source = """
        <html>
            <head>
                <title></title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f0f0f0;
                        text-align: left;
                    }
                    img {
                        max-width: 300px;
                        border: 2px solid #333;
                        border-radius: 10px;
                        box-sizing: border-box;
                    }
                    .container {
                        margin: 0 auto;
                        background-color: #fff;
                        display: flex;
                        align-items: center;
                    }
                    .text {
                        flex: 1;
                        padding: 20px;
                        text-align: justify;
                    }
                    h1 {
                        color: #333;
                        font-weight: normal;
                        text-align: center;
                        white-space: nowrap;
                    }
                </style>
            </head>
            <body>
            <h1>%s</h1>
            <div class="container">
                        
                            <div class="text">

                <p>%s</p>
                
            </body>
        </html>
    """
            answer = result['answer'].replace("\n", "<br>")
            output = "<br> <br>" + answer
            
            final_html = source % (query, output)
            print(f"{white}Answer: " + str(result["answer"]))
            
            with open('static/html.txt', 'w') as f:
                f.write(final_html)
            return result['answer']

        
        else:
            result = pdf_qa(
                {"question": query, "chat_history": chat_history})
            print(f"{white}Answer: " + str(result))
            #print((result['source_documents'][0].metadata)['source'])
            """link = ((result['source_documents'][0].metadata)['source'].split('/')[2])
            link = link.replace('_','/')
            print("Please find the informationa t ",link)"""
            #tab.showWebview(link)
            try:
                page_no = result['source_documents'][0].metadata['page']
                print("PdF -->", result['source_documents'][0].metadata)
                #images1[page_no]
                images1[page_no].save('../static/output_img1.png')
            except:
                pass
            try:
                if 'http' in (result['source_documents'][0].metadata)['source']:
                    website_url = (result['source_documents'][0].metadata)['source'].split('/')[2][:-4]
                    website_url = website_url.replace("[","/")
                    qr = qrcode.QRCode(
                    version=1,  # QR code version (adjust as needed)
                    error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
                    box_size=10,  # Size of each box in the QR code
                    border=4,  # Border size around the QR code
                )
                    qr.add_data(website_url)
                    # Make the QR code
                    qr.make(fit=True)
                    # Create an image from the QR code
                    qr_image = qr.make_image(fill_color="black", back_color="white")
                    qr_image.save("../static/output_img1.png")
            except:
                pass
            chat_history.append((query, result["answer"]))
            source = """
        <html>
            <head>
                <title></title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #F0F0F0;
                        text-align: left;
                    }
                    img {
                        max-width: 300px;
                        border: 2px solid #333;
                        border-radius: 10px;
                        box-sizing: border-box;
                    }
                    .container {
                        margin: 0 auto;
                        background-color: #fff;
                        display: flex;
                        align-items: center;
                    }
                    .text {
                        flex: 1;
                        padding: 20px;
                        text-align: justify;
                    }
                    h1 {
                        color: #333;
                        font-weight: normal;
                        text-align: center;
                        white-space: nowrap;
                    }
                </style>
            </head>
            <body>
            <h1>%s</h1>
            <div class="container">
                            <div class="text">
                <p>%s</p>
                </div>
                <img src="%s">
                </div>
            </body>
        </html>
    """
            answer = result['answer'].replace("\n", "<br>")
            output = "<br> <br>" + answer + " <br><br> Please find the source: <br>"
            final_html = source % (query, output, "../static/output_img1.png")
            with open('../static/html.txt', 'w') as f:
                f.write(final_html)
            return result['answer']