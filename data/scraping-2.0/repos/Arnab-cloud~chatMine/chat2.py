import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import pickle
os.environ["OPENAI_API_KEY"] = "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"

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


# Load the docsearch index from a file using pickle
with open("docsearch_index.pkl", "rb") as file:
    docsearch = pickle.load(file)

chain = load_qa_chain(OpenAI(),chain_type="map_rerank",return_intermediate_steps = True)
query = input("Enter: ")
docs = docsearch.similarity_search(query,k=10)

chain.llm_chain.prompt.template

#chain.run(input_documents=docs, question=query)

results = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
print(results["output_text"])
