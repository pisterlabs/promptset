from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import os
from dotenv import load_dotenv
import json
from article import Article

load_dotenv()

# Load the JSON data
with open('samples/sample-articles.json', 'r') as f:
  data = json.load(f)

# Create Article objects
articles = [Article(**article) for article in data]


documents = [
  Document(
    page_content=str(article),
    metadata=article.to_dict()
  )
  for article in articles
]

embeddings = OpenAIEmbeddings()
# article_embeddings = [embeddings.(article.text) for article in articles]
vectorstore = Chroma.from_documents(documents, embeddings)
# vectorstore = Chroma.from_embeddings(article_embeddings)

def find_best_articles(prompts):
  interesting_articles = {}
  for prompt in prompts:
    docs = vectorstore.similarity_search(prompt, k=5)
    interesting_articles[prompt] = [doc.metadata for doc in docs]
  return interesting_articles

prompts = ["49ers"]
interesting_articles = find_best_articles(prompts)

print('Total articles:', len(articles), '\n\n\n\n')

for prompt, articles in interesting_articles.items():
  print(prompt, '\n')
  for article in articles:
    print(article['title'])
    
    print('\n')
  print('\n\n\n\n')


def generate_prompts():
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate some prompts related to NBA."}
    ]
  )
  return response['choices'][0]['message']['content'].split(', ')

# print(interesting_articles)