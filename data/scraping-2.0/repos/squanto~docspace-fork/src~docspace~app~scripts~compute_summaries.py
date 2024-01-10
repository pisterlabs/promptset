from app import *
import threading
import time
from tqdm import tqdm
import docspace
import cohere

co = cohere.Client(docspace.config['COHERE_API_KEY'])

docs = Document.objects.all()

for doc in tqdm(docs):
    doc.update_chunks(skip_similarity_matching=True)
    




