import os
import requests
import numpy as np
import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# - choose the embeddings Model
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


#get pdf files
os.makedirs("data", exist_ok=True)
files = [
    "https://www.cancer.gov/publications/patient-education/takingtime.pdf",
]

for url in files:
    file_path = os.path.join("data", url.rpartition("/")[2])
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(r.content)
    else:
        print(f"Failed to retrieve {url}")
        
#chunk it up
loader = PyPDFDirectoryLoader("./data/")
documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)

#Look at sample embeddings
#sample_embedding = np.array(hf.embed_query(docs[0].page_content))
#print("Sample embedding of a document chunk: ", sample_embedding)
#print("Size of the embedding: ", sample_embedding.shape)

#create sample embeddings for the entire db

vectorstore_faiss = FAISS.from_documents(
    docs,
    hf,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

query = "What are steps to cancer survival"

#make an embedding of the query
query_embedding = vectorstore_faiss.embedding_function(query)
np.array(query_embedding)

#now find relevant match from the corpus using the query embedding
relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
print('----')
answer = []
for i, rel_doc in enumerate(relevant_documents):
    print(f'## Document {i+1}: {rel_doc.page_content}.......')
    answer.append(rel_doc.page_content)

#concatenate the answers             
answer = ''.join(answer)
    
# Set up your open API key and endpoint
openai.api_key = "here goes you api" #enter your API here

if answer:  # This checks if the answers list is not empty
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"The document says: '{answer}'. What is the main idea?",
        max_tokens=200,
        temperature=1,
    )
    print("ChatGPT Interpretation:")
    print(response.choices[0].text.strip())
else:
    print("No results noted in local query database.")

