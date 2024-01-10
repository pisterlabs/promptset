# %%

"""
Evaluate a RAG pipeline

*Components*
Retriver Component: Retrieves additional context from an external database for the LLM to answer the query
Generator Component: Generates an answer based on a prompt augmented with the retrieved information

*Evaluation Metrics*
BLEU: Bilingual Evaluation Understudy
ROUGE: Recall-Oriented Understudy for Gisting Evaluation
METEOR: Metric for Evaluation of Translation with Explicit ORdering
RAGAs: Retrieval-Augmented Generation Assessment
"""

# %%
# pip install langchain openai weaviate-client ragas

# %%
# 1. prepare the data by loading and chunking the documents

import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Download the data
url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

# Load the data
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

# Chunk the data
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# %%
# 2. generate the vector embeddings for each chunk with the OpenAI embedding model and store them in the vector database.

import chromadb
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

# Load OpenAI API key from .env file
load_dotenv(find_dotenv())

# Setup vector database
chroma_client = chromadb.Client()

# Create a collection
collection = chroma_client.create_collection(name="my_vector_collection")

# Populate vector database
for chunk in chunks:
    # Generate embeddings for each chunk
    embedding = OpenAIEmbeddings().embed(chunk)
    # Add the chunk and its embedding to the collection
    collection.add(
        embeddings=[embedding.tolist()],
        documents=[chunk],
        ids=[str(id(chunk))]
    )

# Define collection as retriever to enable semantic search
# Define a function to perform semantic search
def search(query_text, n_results=10):
    # Generate an embedding for the query text
    query_embedding = OpenAIEmbeddings().embed(query_text)
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results

# Use the search function as the retriever
retriever = search
# %%
# 3. Set up a prompt template and the OpenAI LLM and combine them with the retriever component to a RAG pipeline

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Define LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define prompt template
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Setup RAG pipeline
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

# %%
# 4. Preparing the Evaluation Data

from datasets import Dataset

questions = ["What did the president say about Justice Breyer?", 
             "What did the president say about Intel's CEO?",
             "What did the president say about gun violence?",
            ]
ground_truths = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
                ["The president asked Congress to pass proven measures to reduce gun violence."]]
answers = []
contexts = []

# Inference
for query in questions:
  answers.append(rag_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# %%
# 5. Evaluating the RAG application

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()