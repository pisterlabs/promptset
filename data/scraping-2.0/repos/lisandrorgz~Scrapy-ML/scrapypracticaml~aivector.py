#To get started, get an API Key from OpenAI, and store in a .env file with the following format.
OPENAI_API_KEY='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'
#We will then install three libraries that we'll work with in this video with the following command:
#pip install langchain openai python-dotenv
#Let's start writing some code. We'll bring some imports in, and will calling the load_dotenv() function from python-dotenv. We'll explain these imports soon.

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#Let's now get some text to use in this tutorial. We'll use the State of the Union text that's referenced on Langchain's documentation. You can find it here.
#Add this text file to your local directory, where the code is located. We can then read this in using Langchain's  TextLoader, as below:
loader = TextLoader('state_of_the_union.txt', encoding='utf-8')
documents = loader.load()

print(documents)  # prints the document objects
print(len(documents))  # 1 - we've only read one file/document into the loader

#Once the document is loaded, we are going to use the langchain  RecursiveCharacterTextSplitter object to split this text into chunks.
#Rather than embedding the entire document as a single vector, we split it into chunks that have more specificity than the entire document taken as a whole, and embed each chunk individually.
#Let's write code to chunk the text:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

print(texts)
print(len(texts))

#This outputs the original document as a set of texts, after splitting into 1000-character chunks, as per the parameters to the RecursiveTextSplitter.
#There's also a small overlap between the chunks, to allow a small amount of context to be shared from one chunk to the next - you might want to increase this overlap!
#You can look at the content of the first chunk with the following code:
print(texts[0])

#Now, let's convert our chunks to embeddings (vectors). We can use the OpenAI integrations in Langchain to do this.

#Langchain comes with an OpenAIEmbeddings object that is used to retrieve embeddings from OpenAI for pieces of text. This object will call the embedding API endpoint with the provided text, which will return the vector embedding.
#The following code demonstrates this:

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query('Testing the embedding model')
print(len(vector))  # 1536 dimensions

#The OpenAIEmbeddings object has an embed_query method, which we use to pass in a text query.
#The query is embedded to a 1536-dimensional vector, with each dimension in the vector encoding a specific "concept" about the passed-in chunk of text.
#These vectors can be compared to other vectors using distance metrics, which we'll see later is added as a feature in vector databases such as pgvector and Chroma.
#Let’s now create vectors for the first 5 chunks in the state of the union text.
#Note: When we split the text with the RecursiveTextSplitter, we get Langchain Document objects - these have a page_content property that stores the actual text for the chunk.
#We’ll reference that property in the following list comprehension, passed as an argument to the OpenAIEmbeddings object's embed_documents() function:

doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:5]])

print(len(doc_vectors))  # 5 vectors in the output
print(doc_vectors[0])    # this will output the first chunk's 1539-dimensional vector

#So here, we have the vectors for the first 5 chunks.
#Next, we want to get vectors for all chunks, and store them in the pgvector database.

# PG VECTOR POSTGRESQL DOCKER

"""
pgvector for Storing Embeddings
Firstly, we need to install PostgreSQL, and enable the pgvector extension. To do this, we'll use Docker, and will pull this image.

You can pull the image with the command: docker pull ankane/pgvector
Once pulled, you can start the container with the following command:

docker run --name pgvector-demo -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d ankane/pgvector

In this command, we use the ankane/pgvector image we just pulled to run a container, and we give the container a name, set the POSTGRES_PASSWORD environment variable, and map port 5432 between the container and host.
Verify that this is running with: docker ps
You can now install a GUI tool such as pgAdmin to inspect the database that is running in the container, or else use psql on the command-line. When connecting, you can specify the host as localhost, and the password as whatever you used in the above command - mysecretpassword, in our case.
We will now create a database, and then add the pgvector extension to that database, with the following SQL commands:

CREATE DATABASE vector_db;
CREATE EXTENSION pgvector;
The pgvector extension we're adding is already installed in this container, since we pulled from the pgvector Docker image. If you're not using this image, you will need to install pgvector separately - see the instructions on the Github repository here.
Now, let’s make a connection to PostgreSQL in our Jupyter Notebook.
To do so, we need a few libraries:

!pip install psycopg2-binary pgvector

Once these are installed, we can take our document chunks from the State of the Union text, and embed these and store in the database.
To do so, we can import the PGVector object from langchain.vectorstores, and use its from_documents() function:
"""

from langchain.vectorstores.pgvector import PGVector

CONNECTION_STRING = "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/vector_db"
COLLECTION_NAME = 'state_of_union_vectors'

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

#This code sets the connection string, and the name of the collection, and passes these to the from_documents() function.
#We also pass, as a first argument, the embedding object that will be responsible for generating vectors from the texts, and as a second argument, the chunked texts themselves.
#Once we execute this code, the embeddings will be stored in the database, using pgvector to do so.

#So we can do this similarity check in the vector space, and we can use the following Langchain code to do so:

query = "What did the president say about Russia"
similar = db.similarity_search_with_score(query, k=2)

for doc in similar:
    print(doc, end="\n\n")

#The db object has a similarity_search_with_score() function that takes a query, and optionally the k closest embeddings we want to return from that query.
#So this function will return the chunks that are "most similar" to the query we pass in. Implicitly, it'll embed the query with OpenAI's embedding model, and will then query the database to find the closest vectors in that 1536-dimensional vector-space.
