from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from flask import Flask, request, jsonify
import torch


app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

@app.route('/llm', methods = ['POST'])
def answer():
    body = request.get_json()
    # Define Question
    question = body['prompt']
    # Define Store and Retrieve
    result = rag_pipeline(question)
    print(result)
    return result

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {device}")
    else:
        print("No GPU detected. Using CPU.")
    n_gpu_layers = 35  # Metal set to 1 is enough.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Load Model
    llm = LlamaCpp(model_path="./models/llama-2-7b-chat.ggmlv3.q8_0.bin", n_gpu_layers=n_gpu_layers, n_batch=n_batch, n_ctx=2048, f16_kv=True, callback_manager=callback_manager, verbose=False)
    # Embedding Model Details
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_model = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs={'device': device}, encode_kwargs={'device': device, 'batch_size': 32})
    arr = ['./india.txt', './china.txt']
    # Load Data
    data = []
    for d in arr:
        loader = UnstructuredFileLoader(d)
        data.append(loader.load()[0])
    # Split Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(len(all_splits))
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
    rag_pipeline = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())
    
    # Start
    app.run(host="0.0.0.0", port=5005)
