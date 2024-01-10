from transformers import AutoTokenizer, AutoModel
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 需要 openai key , 並不是真的可以離線
# pip install cpm_kernels
# pip install langchain
# pip install unstructured

os.environ["TORCH_HOME"] = './models'
model_name = 'THUDM/chatglm-6b'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
# model.save_pretrained('models/chatglm-6b-float16')
chatglm = model.eval()


filepath = 'data/vue3.txt'
loader = UnstructuredFileLoader(filepath)
docs = loader.load()


with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()


text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
docs = text_splitter.split_text(text)


embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)


query = "如何建立 vue3 app?"
docs = vector_store.similarity_search(query)
context = [doc.pag_content for doc in docs]

prompt = f"已知訊息:\n{context}\n根據已知訊息回答問題:\n{query}"

chatglm.chat(tokenizer, prompt, history=[])
