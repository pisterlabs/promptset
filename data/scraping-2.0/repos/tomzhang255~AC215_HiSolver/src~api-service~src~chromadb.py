import json
import os
import sys
import asyncio
import contextlib
import socket
import numpy as np
from sklearn.cluster import KMeans
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

# FastAPI and related imports
from fastapi import UploadFile, File, HTTPException, Path, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Langchain and related imports
import langchain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import OpenAI, Anthropic
from langchain.chains import (
    ConversationalRetrievalChain, RetrievalQA, LLMChain, LLMMathChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import AIMessage, HumanMessage, SystemMessage, Document
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# ChromaDB imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Application imports
from .app.app import create_app
from .animation.hintGen import animation_from_question

# Load configuration
with open("config.json") as json_data_file:
    config = json.load(json_data_file)

# Set environment variables
OPENAI_API_KEY = config["OPENAI_API_KEY"]
os.environ["TRANSFORMERS_CACHE"] = "./src/model"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# CORS origins
origins = [
    "http://localhost:8001",  
    "http://localhost"
]

# Create FastAPI app
app = create_app()



# Utility function to run tasks in an executor
async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)



# Digest the -qm flag directly
# Setting the configurations for medium quality
config['pixel_height'] = 720  # Set height in pixels
config['pixel_width'] = 1280  # Set width in pixels
config['frame_rate'] = 30     # Set frames per second


# Endpoint for generating animations from questions
@app.get("/animation_from_question")
async def animation_from_question_endpoint(video_name: str = "test", query: str = ""):
    """
    This endpoint handles requests to generate animations based on a given query.
    The function takes two parameters: 'video_name' and 'query'.
    
    Args:
        video_name (str, optional): The name of the output video file. Defaults to "test".
        query (str, optional): A string containing the question or scenario description 
                               based on which the animation is to be generated. This should be 
                               a detailed description or a question that can be visually represented.
    
    The endpoint uses the 'animation_from_question' function to process the input query 
    and generate a corresponding animation video. The video is saved in the specified 
    'video_path' directory with the given 'video_name'.

    The function returns a FileResponse, allowing the client to download or stream 
    the generated video file directly.

    Returns:
        FileResponse: A response object containing the generated video file. 
                      The media type of the response is set to 'video/mp4'.
    """
    video_path = f"./media/videos/1080p60/"
    # query = '''
    # A high school's math club is creating a large mosaic for their annual fundraiser, using colored tiles to form a right triangle on a square courtyard floor. The length of the triangle's base is 24 feet, and the height is 32 feet. The club plans to fill the entire triangle with tiles, and each square tile measures 1 foot by 1 foot. After laying down the tiles for the triangle, the students decide to add a border made of the same tiles around the triangle. This border will be one tile wide and will run along the two legs of the triangle and the hypotenuse, but not along the base or height. How many tiles will the math club need to create the border?
    # '''
    await run_in_executor(animation_from_question, video_name, query)
    return FileResponse(video_path + video_name, media_type="video/mp4")
