
import os
import redis

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time
from langchain_memorydb import MemoryDB as Redis

# Constants
INDEX_NAME = 'idx:vss-mm'
REDIS_URL = "rediss://clusterurl:6379/ssl=True&ssl_cert_reqs=none"
pdf_path= "memorydb-guide.pdf"
def initialize_redis():
    client = redis.Redis(
        host='clusterurl',
        port=6379, decode_responses=True, ssl=True, ssl_cert_reqs="none")
    try:
        client.ping()
        print("Connection to MemoryDB successful")
        return client
    except Exception as e:
        print("An error occurred while connecting to Redis:", e)
        return None

# Initialize Bedrock model
def get_llm():
    model_kwargs = {"max_tokens_to_sample": 8000,
            "temperature": 0.2, 
            "top_k": 250, 
            "top_p": 0.9,
            "stop_sequences": ["\\n\\nHuman:"]
               }    
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        #model_id="mmeta.llama2-13b-chat-v1",
        model_id="anthropic.claude-instant-v1", #use the Anthropic Claude model
        model_kwargs=model_kwargs
) #configure the properties for Claude

    return llm
# Initialize embeddings
def initialize_embeddings():
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"),
    )
    return embeddings
    
def check_index_existence():
    try:
        client=initialize_redis()
        info = client.ft(INDEX_NAME).info()
        num_docs = info.get('num_docs', 'N/A')
        space_usage = info.get('space_usage', 'N/A')
        num_indexed_vectors = info.get('num_indexed_vectors', 'N/A')
        vector_space_usage = info.get('vector_space_usage', 'N/A')
        index_details = {
            'num_docs': num_docs,
            'space_usage': space_usage,
            'num_indexed_vectors': num_indexed_vectors,
            'vector_space_usage': vector_space_usage,
            'exists': True
        }
        return index_details
    except Exception:
        return {'exists': False}

def initializeVectorStore():
    # Start measuring the execution time of the function
    start_time = time.time()
    embeddings=initialize_embeddings()
    try:
        # Load and split PDF
        # Initialize the PDF loader with the specified file path
        loader = PyPDFLoader(file_path=pdf_path)
        # Load the PDF pages
        pages = loader.load_and_split()
        # Define the text splitter settings for chunking the text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=100
        )
        # Split the text into chunks using the defined splitter
        chunks = loader.load_and_split(text_splitter)

        # Create Redis vector store
        # Initialize the Redis vector store with the chunks and embedding details
        vectorstore = Redis.from_documents(
            chunks,
            embedding=embeddings,
            redis_url=REDIS_URL,
            index_name=INDEX_NAME,
        )

        # Calculate and print the execution time upon successful completion
        end_time = time.time()
        print(f"initializeVectorStore() executed in {end_time - start_time:.2f} seconds")

        return vectorstore

    except Exception as e:
        # Handle any exceptions that occur during execution
        # Calculate and print the execution time till the point of failure
        end_time = time.time()
        print(f"Error occurred during initializeVectorStore(): {e}")
        print(f"Failed execution time: {end_time - start_time:.2f} seconds")
        # Return None to indicate failure
        return None

redis_client = Redis(
            redis_url=REDIS_URL,
            index_name=INDEX_NAME,
            embedding=initialize_embeddings(),
           # index_schema=index_schema  # Include the index schema if provided
        )

def initializeRetriever():
    """
    Initializes a Redis instance as a retriever for an existing vector store.

    :param redis_url: The URL of the Redis instance.
    :param index_name: The name of the index in the Redis vector store.
    :param embeddings: The embeddings to use for the retriever.
    :param index_schema: (Optional) The index schema, if needed.
    :return: The retriever object or None in case of an error.
    """
    index_name=INDEX_NAME
    redis_url=REDIS_URL
    embeddings=initialize_embeddings()
    try:
        # Start measuring time for Redis initialization
        start_time_redis = time.time()

        # Initialize the Redis instance with the given parameters
        
        # Measure and print the time taken for Redis initialization
        end_time_redis = time.time()
        print(f"Vector store initialization time: {(end_time_redis - start_time_redis) * 1000:.2f} ms")

        # Start measuring time for retriever initialization
        start_time_retriever = time.time()

        # Get the retriever from the Redis instance
        retriever = redis_client.as_retriever()

        # Measure and print the time taken for retriever initialization
        end_time_retriever = time.time()
        print(f"Retriever initialization time: {(end_time_retriever - start_time_retriever) * 1000:.2f} ms")

        return retriever

    except Exception as e:
        # Print the error message in case of an exception
        print(f"Error occurred during initialization: {e}")
        return None

def perform_query(query):
    results = rds.similarity_search(query)
    return results

# Initialize Retrieval QA with prompt
def query_and_get_response(question):
    prompt_template = """Human: Use the following pieces of context to provide a concise answer in English to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Assistant:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm=get_llm()
    retriever=initializeRetriever()
    qa_prompt = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose = True ,
    )

    result = qa_prompt({"query": question})
    return result["result"]
def noContext(question):
    llm = get_llm()
    
    # Construct a prompt that instructs the LLM to provide concise answers
    concise_prompt = "Please provide a concise answer to the following question:\n\n"
    # Combine the concise instruction with the user's question
    full_question = concise_prompt + question

    try:
        # Generate a response using the LLM
        response_text = llm.predict(full_question)  # Pass the combined prompt and question to the model
        return response_text
    except Exception as e:
        # Handle any exceptions that occur during LLM prediction
        print(f"Error during LLM prediction: {e}")
        return None
