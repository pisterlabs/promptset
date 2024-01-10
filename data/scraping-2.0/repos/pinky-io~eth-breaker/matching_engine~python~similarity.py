from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import json
import csv
import os
import requests
model = SentenceTransformer('all-MiniLM-L6-v2')



print(community)