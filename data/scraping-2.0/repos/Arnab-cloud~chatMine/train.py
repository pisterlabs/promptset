import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import pickle
os.environ["OPENAI_API_KEY"] = "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"
'''
doc_reader = PdfReader("the_explosives_act_1884_2.pdf")

raw_text = ""
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

'''
# List of PDF files to process
pdf_files = ["the_explosives_act_1884_2.pdf", "The_Mines_Act_1952.pdf", "Colliery_control_order_0_0.pdf"]

# Initialize an empty string to store combined text
raw_text = ""

# Iterate through PDF files and extract text
for pdf_file in pdf_files:
    doc_reader = PdfReader(pdf_file)
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
            
text_splitter = CharacterTextSplitter(
    separator= "\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function= len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts,embeddings)

# Save the docsearch index to a file using pickle
with open("docsearch_index.pkl", "wb") as file:
    pickle.dump(docsearch, file)
print("Okk..")

