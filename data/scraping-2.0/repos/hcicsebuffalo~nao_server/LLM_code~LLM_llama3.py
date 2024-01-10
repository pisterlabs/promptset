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

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
os.environ["TRANSFORMERS_CACHE"] = '/data/'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
images1 = convert_from_bytes(open(
    'data/grad-handbook-2023.pdf', 'rb').read())

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
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
documents = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")
vectordb = FAISS.from_documents(documents, embeddings)
try:
    print("Entered context handbook updation")
    with open('../context_handbook.txt', 'r') as file:
        data = file.read().replace('\n', '')
except:
    print("No data in context_handbook")
try:
    vectordb.add_texts([str(data)])
except:
    print("Context handbook added to vectordb")
    
from torch import cuda, bfloat16
import transformers
#Deci/DeciLM-6b-instruct
#meta-llama/Llama-2-13b-chat-hf
model_id = 'meta-llama/Llama-2-13b-chat-hf'
#model_id = 'TheBloke/Llama-2-7B-Chat-GPTQ'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory

# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config, 
    #quantization_config=bnb_config,
    #device="cuda:0",
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    #stopping_criteria=stopping_criteria,  # without this model rambles during chat
    do_sample=True,
    temperature=0.2,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=4096,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

pdf_qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_type = "similarity_score_threshold", search_kwargs={'score_threshold': 0.5, 'k': 4}),
    max_tokens_limit=4000,
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
            chain = ConversationalRetrievalChain.from_llm(llm,
                                                          vectordb.as_retriever(search_kwargs={"k": 2}),verbose=False)
            result['answer'] = "The information is updated.Thank you"
            
            with open("../context_handbook.txt", "a") as context_file:
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
            
            with open('../static/html.txt', 'w') as f:
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