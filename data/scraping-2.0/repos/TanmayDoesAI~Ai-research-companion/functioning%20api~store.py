import json
import pickle
import pinecone
from tqdm.auto import tqdm, trange
from glob import glob
from pdf.read_pdf import get_pdf_content, get_pdf_content_from_bytes
from qa.generate_embeddings import OpenAIEmbeddings
from db.save_embeddings import PineconeDB

from config.config import *

docs=["C:/Users/tanma/OneDrive/Desktop/2309.07906.pdf"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


flatten = lambda lst: [item for sublist in lst for item in sublist]

openai_embeddings = OpenAIEmbeddings()

pdf_contents = []

for _, pdf in enumerate(tqdm(docs, desc="Getting PDFs")):
    pdf_contents.append(get_pdf_content(pdf))

embeddings = openai_embeddings(flatten(pdf_contents), 512)

with open("./pdf_embeddings.pkl", "wb") as fp:
    pickle.dump(embeddings, fp)

batch_size = 50

db = PineconeDB(PINECONE_API_KEY, PINECONE_ENV, "pdf", "transformer-papers")
db(embeddings)
