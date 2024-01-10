import json
import boto3
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator

loader = PyPDFLoader("/resource/211122_amlcft_guidelines.pdf")
