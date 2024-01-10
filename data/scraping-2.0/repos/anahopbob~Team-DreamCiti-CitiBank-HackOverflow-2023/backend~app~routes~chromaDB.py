from fastapi import APIRouter, Query, File, UploadFile, HTTPException, Response, status, Form
from fastapi.responses import JSONResponse

from app.embeddings.MiniLM_embedder import MiniLM_embedder
from app.services import chroma_service
from app.models import chromaDocument 
from app.services.summary import SummariseContext
from app.embeddings.imageToText import ImageToText
from app.embeddings.imageEmbedding import ImageEmbedding
from app.services.webscrape import WebScrape
from app.routes.mysqlDB import insert_object_excerpt_pairs, insert_object_info

import chromadb
import uuid
import fitz
import PIL.Image
import io
import os
import tabula
import pandas as pd
import boto3
import traceback
import asyncio

from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger

from typing import List, Union

from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text, extract_pages

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain

import urllib.request

Document = chromaDocument.Document
DocumentParser = chroma_service.DocumentParser

ImageDict = chromaDocument.Image
# imageToText = imageToText.ImageToText

router = APIRouter()

client = chromadb.PersistentClient(
    path="db/chroma.db"
)   
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

imagetotext_collection = client.get_or_create_collection(
    name="imagetotext",
    metadata={"hnsw:space": "cosine"}
)

imagesembedding_collection = client.get_or_create_collection(
    name="imageembedding",
    metadata={"hnsw:space": "cosine"}
)

ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
ACCESS_SECRET_KEY = os.getenv("ACCESS_SECRET_KEY")

#initialize S3 client
BUCKET_NAME='dreamciti'

s3Client = boto3.client('s3',
    aws_access_key_id = ACCESS_KEY_ID,
    aws_secret_access_key = ACCESS_SECRET_KEY
)

s3Resource = boto3.resource('s3',
    aws_access_key_id = ACCESS_KEY_ID,
    aws_secret_access_key = ACCESS_SECRET_KEY
)

# <=========================== chromaDB routes ===========================>
# async function for PDF file upload to s3 bucket
async def save_upload_file(file: UploadFile, fileName: str):
    try:
        # Save a local copy of the uploaded file
        with open(fileName, "wb") as local_file:
            while True:
                content = await file.read(1024)
                if not content:
                    break
                local_file.write(content)

        # Upload the local file to S3
        s3Client.upload_file(fileName, BUCKET_NAME, fileName)

        # Remove the local file after uploading to S3
        os.remove(fileName)

        return JSONResponse(content={"message": "Upload successfull"}, status_code=200) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# async function for PDF text and image parse for AI VectorDB enrollment
async def process_pdf_and_enroll(file: UploadFile, fileName: str, department: str) -> dict:
    try:
        file.file.seek(0)

        # Read the PDF file
        pdfContent = await file.read()

        # Create a PDF document object using PyMuPDF (Fitz)
        pdf_document = fitz.open(stream=pdfContent, filetype="pdf")

        # Directory to save extracted images
        images_dir = "images"
        os.makedirs(images_dir, exist_ok=True)

        # Initialize a variable to store all the extracted text, with delimiter for separate pages and images
        extractedText = []

        # Iterate through each page of the PDF
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)

            # Extract text from the page
            pageText = page.get_text()

            # Check if the page contains images and extract all images (using OCR to detect images)
            images = page.get_images(full=True)

            # If the page contains images, add a unique image indicator to the extracted text for that particular page only
            if images:
                for idx, image in enumerate(images):
                    xref = image[0]
                    base_img = pdf_document.extract_image(xref)
                    image_data = base_img["image"]
                    extension = base_img["ext"]

                    # Renaming the image file accordingly with a unique id for text extraction formatting
                    uniqueId = str(uuid.uuid4())[:]
                    image_id = f"<?% type=image,object_id={uniqueId} %>"

                    # (to be linked with S3)
                    # image_name = f"type=image,object_id={uniqueId}.{extension}"

                    # Currently saves to the images folder in the backend root directory (to be changed to S3 bucket)
                    # with open(test_image_id, "wb") as image_file:
                    #     image_file.write(image_data)

                    # Add image indicator to the extracted text
                    pageText = image_id + pageText + image_id

            # Add the extracted text to the overall extracted text
            extractedText.append(pageText)

        # Close the PDF document after iteration completed
        pdf_document.close()

        # Invoke AI PDF Enrollment function here (ID to be retrieved from S3, department to be retrieved from frontend)
        finalText = ''.join(extractedText)

        finalDocument = {
            "id": fileName,
            "text": finalText,
            "department": department
        }

        # Invoke async AI enrollment function here and await response
        AiTask = await enroll(finalDocument)

        # Check the AI enrollment response and return it
        if AiTask.status_code == 200:
            return JSONResponse(content={"message": "Enrollment successfull"}, status_code=200)   
        else:
            return JSONResponse(content={"message": "Error in enrollment"}, status_code=500)

    except Exception as e:
        # Handle any exceptions here
        return {"error": str(e)}
    

@router.post("/pdf-enroll")
async def pdf_enroll(
    file: UploadFile, 
    department: str = Form(...)
):
    """
    Given a file (PDF/Word) upload, extract its text and images. 
    The file and its images will be stored in the AWS S3 bucket and 
    together with the reformatted text, it will then be enrolled into the Vector DB.
    """
    # Extracting file properties
    pdfFile = file.file
    fileName = file.filename
    fileExtension = file.filename.split(".")[-1].lower()

    # Insert into mysqlDB
    object_info = {
        "ObjectID": fileName,
        "ObjectName": fileName,
        "Department": department,
        "Classification": "Restricited",
        "Upvotes": 0,
        "Downvotes": 0,
        "isLink": False,
        "URL": ""
    }
    _ = insert_object_info(object_info)

    # Determining file type
    if fileExtension == "pdf":
        try:
            # Creating an async task for the S3 upload and run it concurrently
            s3_upload_task = asyncio.create_task(save_upload_file(file, fileName))

            # Seek back to the beginning of the uploaded file
            file.file.seek(0)

            # Creating an asyncio task for AI enrollment
            AiTask = asyncio.create_task(process_pdf_and_enroll(file, fileName, department))

            # Await both responses
            s3Response = await s3_upload_task
            AIresponse = await AiTask

            if AIresponse.status_code == 200 and s3Response.status_code == 200:
                return {"message": "Successfully uploaded PDF"}
            else:
                return {"message": "Error in uploading PDF"}
        
        # catching and returning any errors
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}
    
    # raising HTTP exception if file is not a PDF
    else:
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF")
    

@router.post("/enroll")
async def enroll(document: Document):
    """
    Given a particular department and text, get embeddings for the text 
    and enroll inside the DB.
    """
    try:
        # Getting info from POST
        id = document.get("id","")
        text = document.get("text","")
        department = document.get("department","")

        # Parses text here, fixes the tags
        texts, content_ids = DocumentParser.parse_raw_texts(text)

        # ============ Start AI Portion==============
        # Get embeddings
        custom_embeddings = MiniLM_embedder()
        embeddings = custom_embeddings(texts)

        # Function to get content_id (replace with your actual logic)

        # Create metadata list
        metadata = [
            {
                "department": department,
                "object_id": id,
                "content_id": content_ids[idx]
            }
            for idx, _ in enumerate(range(len(texts)))
        ]
           # Generating unique IDs for each document
        excerpt_ids = [str(uuid.uuid4()) for x in range(len(texts))]

        # Associating the object_id with the excerpt_id (using a NoSQL way)
        id_pairing = {
            "ObjectID" : id,
            "ExcerptIDs" : excerpt_ids
        }
        # Inserting into MySQL
  
        _ = insert_object_excerpt_pairs(id,excerpt_ids)

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata,
            ids=excerpt_ids # Generated by us uuid4.uuid()
        )
        # ============Start AI Portion==============
        return JSONResponse(content={"message": "Successfully enrolled"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)


@router.post("/get-caption")
def enroll_image_caption(file: UploadFile):
    """
    Given a particular department and image, get embeddings for the image 
    and enroll inside the DB.
    """
    imgFile = file.file
    fileName = file.filename
    # get text of the image
    image_obj = ImageToText()
    print(fileName)
    abs_path = "C:\\Users\\yimen\\Documents\\GitHub\\Team-DreamCiti-CitiBank-HackOverflow-2023\\frontend\\src\\images"
    img_path = os.path.join(abs_path, fileName)
    image_text = image_obj.getImageToText(img_path)

    return image_text
    



# @router.post("/enroll-image-caption")
# def enroll_image_caption(image: ImageDict):
#     """
#     Given a particular department and image, get embeddings for the image 
#     and enroll inside the DB.
#     """
#     # Getting info from POST
#     image_dict = image.dict()
#     id = image_dict.get("id","")
#     file = image_dict.get("file","")
#     department = image_dict.get("department","")

#     # get text of the image
#     image_obj = ImageToText()
#     if file.startswith('http'):
#         file = urllib.request.urlopen(file)
  
#     image_text = image_obj.getImageToText(file)


#     # ============ Start AI Portion==============
#     # Get embeddings
#     custom_embeddings = MiniLM_embedder()
#     embeddings = custom_embeddings(image_text)

#     # Function to get content_id (replace with your actual logic)

#     # Create metadata list
#     metadata = [
#         {
#             "department": department,
#             "object_id": id,
#             "content_id": idx
#         }
#         for idx, _ in enumerate(range(len(image_text)))
#     ]
#     imagetotext_collection.add(
#         embeddings=embeddings,
#         documents=image_text,
#         metadatas=metadata,
#         ids=[str(uuid.uuid4()) for x in range(len(image_text))] # Generated by us uuid4.uuid()
#     )
#     # ============Start AI Portion==============

#     return None


@router.post("/enroll-image-embedding")
def enroll_image_embedding(image: ImageDict)->None:
    """
    Given a particular department and image, get embeddings for the image 
    and enroll inside the DB.
    """
    # Getting info from POST
    image_dict = image.dict()
    id = image_dict.get("id","")
    file = image_dict.get("file","")
    department = image_dict.get("department","")

    # get text of the image
    
    # ============ Start AI Portion==============
    # Get embeddings
   
    embeddings = ImageEmbedding.get_image_embeddings(file)

    # Create metadata list
    metadata = [
        {
            "department": department,
            "object_id": id,
            "content_id": idx
        }
        for idx, _ in enumerate(range(len(embeddings)))
    ]


    imagesembedding_collection.add(
        embeddings=embeddings,
        documents=file,
        metadatas=metadata,
        ids=[str(uuid.uuid4()) for x in range(len(embeddings))] # Generated by us uuid4.uuid()
    )
    # ============Start AI Portion==============

    return None


@router.get("/search/")
def search_items(
    department: str = Query(None, description="Department name (optional)"),
    query: str = Query(..., description="Query string"),
):
    # Use 5 Chunks of text to do the similarity search
    if department == None:
        results = collection.query(
            query_texts=[query],
            n_results=5,
        )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=5,
            where={"department": department}
        )

    if len(results) == 0:
        return JSONResponse(content={"message": "No results found"}, status_code=200)
    else:
        return JSONResponse(content={"results": results , "query": query}, status_code=200)


@router.post("/summarise")
def summarise_items(
    results_dict: dict
    ) -> str:
    try:
        summary_output = SummariseContext.summarise_context(results_dict)
        return JSONResponse(content={"summary": summary_output}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)
    
# @router.post("/delete")
# def delete_object_from_chromadb(
#         object_excerpt_list: List[dict]
#     ):
#     """
#     THIS ROUTE HAS BEEN REFACTORED TO BE USED IN mysqlDB.py.
#     IT WILL NOT BE CALLED FROM THE ROUTER ABOVE.
#     Given an object_id that corrosponds to the one inside S3,
#     delete all related embeddings inside ChromaDB.
#     Currently, association is hard coded.
#     """
#     # Connect to  bucket to get association list 
#     if len(object_excerpt_list) == 0:
#         return []
#     try:
#         association = []
#         for pair in object_excerpt_list:
#             association.append(pair["excerpt_id"])
#         collection.delete(
#             ids=association
#         )
#         return len(association)
#     except Exception as e:
#         return None

# =============== Image Related Endpoints =================


@router.post("/webscrape") # input: {"website": "https://www.google.com/"}
async def get_webscrape(
    website_dict: dict,  
):
    
    website_id = str(uuid.uuid4())
    website = website_dict["website"]
    department = website_dict["department"]
    results = WebScrape.getWebScrape(website)
    document = {"id": website_id,
                 "text": results, 
                 "department": department
    }

    # Insert into mysqldb
    object_info = {
        "ObjectID": website_id,
        "ObjectName": website,
        "Department": department,
        "Classification": "Restricted",
        "Upvotes": 0,
        "Downvotes": 0,
        "isLink": True,
        "URL": website
    }

    insert_object_info(object_info)

    # Invoke async AI enrollment function here and await response
    webScrapeTask = await enroll(document)
    # Check the AI enrollment response and return it
    if webScrapeTask.status_code == 200:
        return JSONResponse(content={"message": "Enrollment successfull"}, status_code=200)   
    else:
        return JSONResponse(content={"message": "Error in enrollment"}, status_code=500)
