from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
# import argparse
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

app = FastAPI()

# Initialize Langchain components outside of the route handler
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
callbacks = [StreamingStdOutCallbackHandler()]

match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    case _default:
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.get("/")
async def root():
    return {"message": "Hello"}

# @app.get("/ask")
# async def ask_question(query: str):
#     start = time.time()
#     res = qa(query)
#     answer = res.get('result', 'No answer available')
#     docs = res.get('source_documents', [])

#     end = time.time()

#     response = {
#         "question": query,
#         "answer": answer,
#         "response_time": round(end - start, 2),
#         "source_documents": []
#     }

#     for document in docs:
#         source_doc = {
#             "source": document.metadata["source"],
#             "page_content": document.page_content
#         }
#         response["source_documents"].append(source_doc)

#     return response

origins = ["http://localhost:3000"]  # Add your React app's origin here

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can specify specific HTTP methods (e.g., ["GET", "POST"])
    allow_headers=["*"],  # You can specify specific headers (e.g., ["Authorization"])
)

@app.post("/ask")
async def ask_question(request: Request):
    # Parse the request body as JSON
    request_body = await request.json()
    
    # Get the query from the request body
    query = request_body.get("query", "")

    start = time.time()
    # Assuming that the 'qa' function takes the query as an argument
    res = qa(query)
    answer = res.get('result', 'No answer available')
    docs = res.get('source_documents', [])

    end = time.time()

    response = {
        "question": query,
        "answer": answer,
        "response_time": round(end - start, 2),
        "source_documents": []
    }

    for document in docs:
        source_doc = {
            "source": document.metadata.get("source", ""),
            "page_content": document.page_content
        }
        response.source_documents.append(source_doc)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
