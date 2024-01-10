from openai import OpenAI

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm
from tqdm.auto import tqdm  # this is our progress bar
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()
# Set up the Pinecone vector database
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = OpenAI()
model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1-aws"
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)
index_name = "doc-prod-test"
index = pinecone.Index(index_name)
query = "A person at my organization made himself in charge.  I am a member of that organization there and I want to file a complaint against the person.  I think he has engaged in lies, stealing money and other things."# Create an embedding for the query
res = client.embeddings.create(input=[query],
model=model_name)

if res and res.data:
    embeddings = res.data[0].embedding
    
similar_docs = index.query(
    vector=embeddings,
    top_k=5,
    include_metadata=True
)
print(f"similar_docs is {similar_docs}")


# # query = "which city has the highest population in the world?"

# # create the query vector
# xq = model_name.encode(res).tolist()

# # now query
# xc = index.query(xq, top_k=5, include_metadata=True)
# xc




# all_embeddings = index.retrieve(list(range(index.info().item_count)))

# # Compare the query embedding with all the embeddings
# query_embedding = res #YOUR_QUERY_EMBEDDING  # Replace with your query embedding
# for embedding in all_embeddings:
#     similarity = pinecone.util.cosine_distance(query_embedding, embedding)
#     print(f"Similarity: {similarity}")

# # Close the connection to the Pinecone vector database
# pinecone.deinit()

# q_res = index.query(xq, top_k=5, include_metadata=True)
# xc


# Retrieve the query embedding vector
# query_embedding = res['data'][0]['embedding']
# # Print the embedded query
# print("Embedded Query:")
# print(query_embedding)

# In August 2001, the appellants filed suit against the appellees,1 alleging that Annette Phelps and Darrin Phelps had wrongfully assumed the positions of co- pastors of the church, of which the appellants averred that they were members. The complaint included causes of action for misrepresentation, embezzlement, defamation, and interference with religious practices. The appellants also purported to demand, on behalf of the church, an accounting of all funds and property of the church, as well as reimbursement for allegedly converted funds.



# # Perform a semantic search

# query_embedding = model.encode([query])[0]
# results = index.query(queries=[query_embedding], top_k=5)

# print(f'These are the results: {results}')

# # Close the Pinecone index
# index.delete()
# pinecone.deinit()
# docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name) #this uploads embeddings and other data to Pinecone

# def initialize_vector_database():
#     # Initialize and return the Pinecone index or other vector database connection
#    
#     index_name = "doc-prod-test"
#     index = pinecone.Index(index_name)
#     return index

# def get_text_embedding(text):
# # Convert text to a vector using OpenAI's embeddings or a similar service
# # Example using OpenAI's GPT-3 model:
#     response = openai.Embedding.create(
#         input=[text],
#         engine="text-similarity-babbage-001"
#     )
#     return response['data'][0]['embedding']

# def search_similar_cases(user_text, vector_database, top_k=5):
#     # Get the vector for the user's text
#     user_vector = get_text_embedding(user_text)

#     # Search in the vector database for similar vectors
#     # Example query in Pinecone:
#     similar_cases = vector_database.query(
#         user_vector, top_k=top_k, include_metadata=True
#     )

#     return similar_cases

# # Main Function
# def main():
#     # User input: description of the legal matter
#     user_input = input("Please describe the facts of your legal matter: ")

#     # Initialize the vector database
#     vector_db = initialize_vector_database()

#     # Search for similar cases
#     similar_cases = search_similar_cases(user_input, vector_db)

#     # Display the results
#     for case in similar_cases:
#         print("Case Text:", case['text'])
#         print("Metadata:", case['metadata'])
#         print("Similarity Score:", case['score'])
#         print()

# if __name__ == "__main__":
#     main()
