import random
import re
import torch
import pdfplumber
import pickle
from flask import Flask, request
from flask_cors import CORS
from haystack.pipelines import GenerativeQAPipeline
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import Seq2SeqGenerator
from haystack.document_stores import PineconeDocumentStore

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BertTokenizerFast, BigBirdPegasusForConditionalGeneration, AutoTokenizer, pipeline