import cloudpickle

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import os

os.environ["openai_api_key"] = "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"

loader = CSVLoader(file_path='test_dataset.csv')

index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

# Save the retriever
with open('retriever.pkl', 'wb') as f:
    cloudpickle.dump(docsearch.vectorstore.as_retriever(), f)

# Save the chain
with open('chain.pkl', 'wb') as f:
    cloudpickle.dump(RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, input_key="question"), f)

# Load the retriever
with open('retriever.pkl', 'rb') as f:
    retriever = cloudpickle.load(f)

# Load the chain
with open('chain.pkl', 'rb') as f:
    chain = cloudpickle.load(f)

query=input("Enter your query: ")
text="tell me  to opt for and some detials of the lawyer and why he/she will be best for the case, give 1 male and 1 female lawyer for case in equal gender proportion"
response =  chain({"question":query+text})

print(response['result'])
