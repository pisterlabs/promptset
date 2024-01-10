"""
記事情報ベクトル化エリア
"""
import glob

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
import sys

def save(text,path):
    with open(path, "w") as file:
        file.write(text)
def read(path):
    with open(path, "r") as file:
        return file.read()

os.environ["PROJECT_ROOT"] = '/Users/nishitsujiyouhei/Documents/RPA/input_and_study/liny-manual-chatbot'
os.environ["CHROME_DRIVER_PATH"] = '/Users/nishitsujiyouhei/Documents/RPA/input_and_study/liny-manual-chatbot/chromedriver'
os.chdir(os.getenv('PROJECT_ROOT'))
import pickle
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
dir_path = './storage/make/help'
documents = []
out_doc = ''
in_doc = ''
base_path = './storage/make/help'
for filepath in glob.glob('./storage/make/help/*/*.txt'):
    filenaame = filepath.replace(base_path+'/','')
    loader = TextLoader(filepath)
    doc = loader.load()[0]
    url = doc.page_content.split('\n')[0].replace('url:','')
    paragraphs = re.sub(r'((?:大|中|小)見出し：(?:[^\n])+?)\n(?=(?:[大|中|小]見出し)：|$)',"", doc.page_content, re.DOTALL)
    paragraphs = re.findall(r'[大|中|小]見出し：(.+?\n.+?)(?=(?:[大|中|小]見出し)：|$)', paragraphs, re.DOTALL)
    for parag in paragraphs:
        # if len(parag)<200 or re.match(r'^.*(?:\n.*){9,}$', parag):
        #     out_doc = out_doc + parag + '\n---\n'
        #     continue
        in_doc = in_doc + parag + '\n---\n'
        documents.append(Document(page_content=parag,metadata={'source':url,'title':filenaame}))
save(out_doc,'./storage/make/help_dash/除外段落.txt')
save(in_doc,'./storage/make/help_dash/適用段落.txt')

# Load Data to vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)


# Save vectorstore
with open("./storage/make/help_dash/vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)