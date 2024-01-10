import os

from global_var import chunk, overlap
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with open('../all_txt.txt', 'r') as file:
    text = file.read()

# Split text into words
python_splitter = PythonCodeTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
docs = python_splitter.create_documents([text])
splitted_text = python_splitter.split_text(text)

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, docs):
        return self.model.encode(docs)

#store = Chroma.from_documents(docs, embeddings, persist_directory='db')
db = Chroma.from_documents(documents=docs, embedding=model, persist_directory='HuggingFaceDB')

db.persist()


prompt = 'Bis wann muss ich die Fragen in Beandworten, multiplechoice fragen in StudIP beantworten?'

search = db.similarity_search_with_score(prompt)

search.sort(key=lambda x: x[1], reverse=True)

for i in search:
    print("Content:" + i[0].page_content)
    print("Relevance:" + str(i[1]))
    print('------------------')



