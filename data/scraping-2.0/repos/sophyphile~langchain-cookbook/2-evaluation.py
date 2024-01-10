# Evaluation
# The process of doing quality checks on the output of your applications. Normal, deterministic code has tests we can run, but judging the output of LLMs is more difficult because of the unpredictability and variability of natural language. LangChain provides tools that aid us on this journey.
# Important to test pipelines regularly to ensure no regression.

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Embeddings, store and retrieval:
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Model and doc loader
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Eval:
from langchain.evaluation.qa import QAEvalChain

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Our long essay from before
loader = TextLoader('data/pgessay.txt')
doc = loader.load()

print (f"You have {len(doc)} document(s)")
print (f"You have {len(doc[0].page_content)} characters in that document")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)

# Get the total number of characters so we can see the average later
num_total_characters = sum([len(x.page_content) for x in docs])

print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")

# Embeddings and docstore
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_documents(docs, embeddings)

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), input_key="question")

question_answers = [
    {'question' : "which company sold the microcomputer kit that his friend built himself?", 'answer' : 'Healthkit'},
    {'question' : "What was the small city he talked about in the city that is the financial capital of USA?", 'answer' : 'Yorkville'}
]

predictions = chain.apply(question_answers)
print(predictions)

# Start your eval chain
eval_chain = QAEvalChain.from_llm(llm)

# Have it grade itself. The code below helps the eval_chain know where the different parts are
graded_outputs = eval_chain.evaluate(question_answers,
                                     predictions,
                                     question_key="question",
                                     prediction_key="result",
                                     answer_key='answer')
print (graded_outputs)


