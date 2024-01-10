from fastapi import FastAPI, UploadFile, File, WebSocket
from starlette.responses import FileResponse
from pydantic import BaseModel
import os
from langchain.document_loaders import PubMedLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
from langchain.embeddings import HuggingFaceEmbeddings
from TextGen import TextGen
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import langchain
from langchain.retrievers import PubMedRetriever
from uuid import uuid4
import shutil
from pathlib import Path
from fastapi import Query
from fastapi.responses import StreamingResponse
import json
from sse_starlette.sse import EventSourceResponse


Prompt_Template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: As if you were a expert in the biomedical domain, answer the question below, only using the context provided to you. Be confident in your answer, do not suggest that you are unsure of your answer. If you do not know the answer, you can say that you do not know.
Please don't say things like this: "Please note that this information should not be used as medical advice or guidance. Always consult with your healthcare provider regarding any questions about medications or treatments for specific conditions."
Context: {context}
Question: {question}

### Response:
"""

Prompt_Template_NousHermes = """### Instruction: 

As if you were a expert in the biomedical domain, answer the question below, only using the context provided to you. Be confident in your answer, do not suggest that you are unsure of your answer. If you do not know the answer, you can say that you do not know. Do no exceed 5 sentences in your answer.
Please don't say things like this: "Please note that this information should not be used as medical advice or guidance. Always consult with your healthcare provider regarding any questions about medications or treatments for specific conditions."
Context: {context}
Question: {question}

### Response:"""

model_url = "wss://mess-scheduled-republican-supports.trycloudflare.com"

llm = TextGen(model_url=model_url, 
                streaming=True,
                max_new_tokens=1024,
                callbacks=[StreamingStdOutCallbackHandler()])

def load_and_split_document(file_path):
    # Load document if file path is provided
    if file_path is not None:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        for page in pages:
            print(page)
            print("-----")
    return pages

def create_vectorstore(texts):
    hf = HuggingFaceEmbeddings(
    model_name='pritamdeka/S-Bluebert-snli-multinli-stsb',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True})
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, hf)
    #retriever = db.as_retriever()
    
    return db

def retrieve_from_pubmed(query):
    pubmed_loader = PubMedLoader(query=query,
                                    load_max_docs=5)
    docs = pubmed_loader.load()
    articles = [doc.page_content for doc in docs if doc.page_content not in [None, '']]
    return articles

def generate_response(retriever, llm, question, external_sources):
    if external_sources == True:
        pubmed_articles = retrieve_from_pubmed(question)
        context = articles[:1]

        results = retriever.similarity_search_with_score(question)[:2]
        for result in results:
            doc, score = result

    else:
        results = retriever.similarity_search_with_score(question)[:2]
        docs = []

        for result in results:
            doc, score = result
            docs.append(doc)
        context = [doc.page_content for doc in docs]
        prompt = Prompt_Template_NousHermes.format(context=context, question=question)
    
    #for response in qa.run(context=context, question=question):
    for response in llm._stream(prompt=prompt):
        # Convert the GenerationChunk to a string
        response_str = response.text  # Replace this with the correct conversion method
        yield response_str  # Yield the string
    
    yield json.dumps({"message": "streaming finished"})

app = FastAPI()

vectorstores = {}

def generate_unique_id():
    # This function generates a unique ID using Python's built-in uuid module.
    # You can replace this with your own function if you want.
    return str(uuid4())


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Define the directory where you want to save the file
    directory = Path("/mnt/c/ari_chain")  # Replace with your directory

    # Create the directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)

    # Create the full path to the file
    file_path = directory / file.filename

    # Save the uploaded file to the directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and split the document
    texts = load_and_split_document(str(file_path))
    id = generate_unique_id()  # You would need to define this function
    vectorstores[id] = create_vectorstore(texts)
    return {"id": id}

@app.get("/answer_question/{id}")
async def answer_question(id: str, question: str = Query(...), external_sources: bool = Query(...)):
    if id in vectorstores:
        return StreamingResponse(generate_response(vectorstores[id], llm, question, external_sources))
    else:
        return {"error": "No PDF with that ID was uploaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
