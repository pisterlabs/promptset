from fastapi import FastAPI, Request
import os
from fastapi import APIRouter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import TokenTextSplitter
from fastapi import UploadFile, File, Form
import shutil
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
import traceback
import datetime
load_dotenv()


router = APIRouter()





@router.post("/trainpdf")
async def train(file: UploadFile = File(...), data_id: str = Form(...)):
    try:
        persist_directory = f'trained_db/{data_id}/{data_id}_all_embeddings'
        
        if file:
            filename = file.filename.lower()

            if filename.endswith(".pdf"):
                pdf_folder_path = f"pdf_temp_{data_id}"
                os.makedirs(pdf_folder_path, exist_ok=True)
                file_path = os.path.join(pdf_folder_path, filename)

                with open(file_path, "wb") as f:
                    f.write(await file.read())

                try:
                    documents = UnstructuredPDFLoader(file_path).load()
                    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
                    split_docs = text_splitter.split_documents(documents)
                    embeddings = OpenAIEmbeddings()

                    new_vectordb = FAISS.from_documents(split_docs, embeddings)
                    try:
                        old_vectordb = FAISS.load_local(persist_directory, embeddings)
                        old_vectordb.merge_from(new_vectordb)
                        old_vectordb.save_local(persist_directory)
                        return {'message':" Embeddings Generated Sucessfully."}


                    except:
                        new_vectordb.save_local(persist_directory)
                        return {'message':" New Embeddings Generated Sucessfully."}



                except Exception as e:
                    # Log the traceback information to a file
                    error_message = f"Error processing PDF: {str(e)}"
                    traceback_str = traceback.format_exc()
                    log_error(data_id, error_message, traceback_str)
                    return {'message': "Error processing PDF Check logs for details"}

                finally:
                    shutil.rmtree(pdf_folder_path)

        else:
            return {'error': "ONLY PDF FILE ALLOWED FOR NOW"}
    
    except Exception as e:
        # Log the traceback information for the general error
        error_message = f"General Error: {str(e)}"
        traceback_str = traceback.format_exc()
        log_error(data_id, error_message, traceback_str)
        return {'error': 'An error occurred. Please check the logs.'}

def log_error(data_id, error_message, traceback_str):
    error_folder = "train_error_logs"
    os.makedirs(error_folder, exist_ok=True)
    error_log_path = os.path.join(error_folder, f"error_log_{data_id}.txt")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_log_path, "a") as error_log:
        error_log.write(f"[{timestamp}] {error_message}\n")
        error_log.write(traceback_str + "\n")



@router.post('/delete')
async def deletetxt(request: Request, data: dict):
    data_id = data.get('data_id')
    delete_directory = f'trained_db/{data_id}'

    # Check if the directory exists
    if os.path.exists(delete_directory):
        # Delete the directory and its contents
        shutil.rmtree(delete_directory)
        return {'message': f"All Embeddings for data_id : {data_id} deleted successfully."}
     
    else:
        return {'message': f"No embeddings found for data id : {data_id} "}
    


