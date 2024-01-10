import os

import arxiv
import pinecone
import openai
import numpy as np

_TOPIC = "Quantum Computing"
_TOPIC_KEY = _TOPIC.replace(" ", "_")

openai.api_key = os.environ['OPENAI_API_KEY']

pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
			  environment=os.environ["PINECONE_ENVIRONMENT"])
index = pinecone.Index("arxiv-index-v2")

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Open the file in write mode
search = arxiv.Search(
	query=_TOPIC,
	max_results=50,
	sort_by=arxiv.SortCriterion.SubmittedDate
)

data_arr = []
try:
	for i, result in enumerate(search.results()):
		try:
			print(f"Ingesting result {i}")
			authors = ','.join([x.name for x in result.authors])
			data = f'Title: {result.title}\nAuthors: {authors}\nSummary: {result.summary}\nPublished On: {result.published}\nCategory: {result.primary_category}\n\n'
			embedding = get_embedding(data)
			data_arr.append((str(i)+_TOPIC_KEY, embedding, {"title": result.title, "url": result.pdf_url, "authors": authors, "summary": result.summary}))
		except Exception as e:
			print(f"Error processing result {i}: {e}")
			continue
except Exception as ea:
	print(f"Error processing arxiv result: {ea}")

index.upsert(vectors=data_arr)