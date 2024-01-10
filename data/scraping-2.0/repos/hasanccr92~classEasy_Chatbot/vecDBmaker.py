from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2TokenizerFast
from pdfTOtxt import make_text

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

'''put the pdf name here'''
make_text('merged.pdf')

text = "temp.txt"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)


chunks = text_splitter.create_documents([text])
embeddings = OpenAIEmbeddings()
vectorstore_openai = FAISS.from_documents(chunks, embeddings)


with open('vector_data.pkl', 'wb') as f:
    pickle.dump(vectorstore_openai, f)


os.remove('./temp.txt')
