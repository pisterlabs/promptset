from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import pickle as pkl
import json

import os
from dotenv import load_dotenv

# Load the environment variable specifying the current environment

# current_environment = os.environ.get('ENVIRONMENT', 'local')

# # Determine the appropriate .env file based on the current environment
# dotenv_file = f'.env.{current_environment}'

# # Load environment variables from the selected .env file
# load_dotenv(dotenv_file)

"""
1. Fetch Relevant Source
2. Setup texts
3. Setup Chain
4. Return generated answer
"""

def generateQuestion(source, query):
    return f"What does {source} say about {query}?"

def generateInsights(question=""):
    OPEN_AI_KEY = os.environ['OPEN_AI_KEY']
    if question == "":
        return []
    vectorstore = FAISS.load_local("vectorstores/faiss_index_clean_code_spacy_splitter", embeddings=OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY))

    retriever = vectorstore.as_retriever()
    template = \
    """
    Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    
    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER : 
    SOURCES: """
    prompt_template = PromptTemplate(template=template, input_variables=["summaries", "question"])
    chain_type_kwargs = {"prompt": prompt_template}

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=OPEN_AI_KEY),
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    answer = chain({"question": question})
    
    with open("saved_answer.pkl", "wb") as f:
        pkl.dump(answer, f)

    return dict(answer)


if __name__ == "__main__":
    query = input("What does Clean Code say about ____\n")
    question = generateQuestion(source="Clean Code", query=query)
    print(generateInsights(question))