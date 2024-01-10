import openai
from uuid import uuid4
from langchain.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
import re
import ast
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from youtube_transcript_api import YouTubeTranscriptApi
import os
import pandas as pd
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub, LLMChain, PromptTemplate

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get('HUGGINGFACE_API_KEY')

app = FastAPI()

class Doubts:
    def __init__(self,userID,username,query,video_link):
        self.query = query
        self.video_link=video_link
        self.username=username
        self.context=None

    def update_variables(self, query=None, video_link=None,username=None):
        if query:
            self.query = query
        if video_link:
            self.video_link = video_link
        if username:
            self.username = username

    def get_transcript(self):
        try:
            video_id = self.video_link.split("=")[1]
            print(video_id)
        except:
            pass
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            result = ""
            for i in transcript:
                result += ' ' + i['text']
            #print(result)
            return result
        except:
            return " "

        # print(transcript)

    def query_context(self):
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        text=self.get_transcript()
        # print(text)
        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        # Step 4: Split text into chunks
        # try:
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 456,
            chunk_overlap  = 24,
            length_function = count_tokens,
        )
        # Get embedding model
        embeddings = HuggingFaceEmbeddings()
        # Create vector database
        chunks = text_splitter.create_documents([text])
        db = FAISS.from_documents(chunks, embeddings)
        docs = db.similarity_search(self.query)
        context=str(docs[0])
        self.context=context if context is not None else " "
        # except:
        

        template=PromptTemplate.from_template("""
                You are Rambaan, a professional Doubts solver for learners. Your job is to solve the doubts of the learner. There is context which is related to the learner's doubt. Use it to provide better answers. 
                Here are the rules that you have to follow:
                - Start by introducing yourself and appreciate {username} for learning this course and encourage the student to learn more.
                - 'Context' section contains the information about the topic of query. Use it to solve the query of learner.
                - Give learner explanations, examples, and analogies about the concept to help them understand and solve their query.
                - Try to answer the question using the context and do not give wrong answers.
                Context:{context}
                Question: {query}
            """)
        
        llm_chain = LLMChain(
            llm=HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature":0.0, "max_length":1024}),
            prompt=template,
            verbose=True,
        )
        # llm2=HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature":0.4, "max_length":1024})
        result = llm_chain.predict(username=self.username, context=self.context, query=self.query)

        return result  
    
class DoubtRequest(BaseModel):
    userId: str
    user_name: str = None
    user_query: str =None
    video_link: str = None

@app.post("/doubts")
def Doubt_Solver(request: DoubtRequest):
    user_name=request.user_name
    user_query=request.user_query
    user_ID=request.userId
    video_link=request.video_link
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    INST=Doubts(user_ID,user_name,user_query,video_link)
    output=INST.query_context()
    return JSONResponse({"data": {"output":output},"status": 200, "message": "Response successful"})