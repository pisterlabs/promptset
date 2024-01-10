from HuggingFaceEmbedding.old.global_var import chunk, overlap
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma
import torch

# Initialize the Sentence-BERT model
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

with open('../all_txt.txt', 'r') as file:
    text = file.read()

# Split text into words
python_splitter = PythonCodeTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
docs = python_splitter.create_documents([text])
splitted_text = python_splitter.split_text(text)


class HuggingFaceEmbeddings:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
        return embeddings

# Initialize the Sentence-BERT model
embeddings = HuggingFaceEmbeddings("sentence-transformers/paraphrase-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings, persist_directory='db')



db.persist()


db = Chroma.from_documents(docs, embedding=embeddings, persist_directory='db')

db.persist()
