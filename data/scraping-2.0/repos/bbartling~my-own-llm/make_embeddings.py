from langchain.embeddings import LlamaCppEmbeddings
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
import os



#MODEL = "./model/ggml-vicuna-7b-1.1-q4_1.bin"
#MODEL = "./model/llama-2-7b-chat.ggmlv3.q2_K.bin"
MODEL = "./model/llama-2-7b-chat.ggmlv3.q6_K.bin"

# Get the model name without the file extension
model_name = os.path.splitext(os.path.basename(MODEL))[0]

# Print the model name
print("model_name: \n",model_name)


file_path = "my_data/hvac.txt"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len
)
chunks = text_splitter.split_text(text=content)

store_name = os.path.splitext(os.path.basename(file_path))[0]
print(f"Using store name: {store_name}")

if os.path.exists(f"{store_name}.pkl"):
    with open(f"my_data/{store_name}_{model_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
else:
    embeddings = LlamaCppEmbeddings(model_path=MODEL)
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"my_data/{store_name}_{model_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)
        
print("Done")