import chromadb
from uuid import uuid4
from utils.pdf_to_text import get_text
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory.chat_message_histories.redis import RedisChatMessageHistory
from services.common import openai_api_key, redis_conn, TTL, context_chat

# This file defines functions for handling knowledge transfer queries by utilizing ChromaDB for storage,
# And OpenAI for text embeddings, and Redis for maintaining chat history

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./kt-db")

# Initialize OpenAI's embedding function
openai_ef = OpenAIEmbeddingFunction(
  api_key=openai_api_key,
  model_name="text-embedding-ada-002"
)

# Intialize LangChain's text splitter for creating smaller chunks from text data
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 10000,
  chunk_overlap  = 20,
  length_function = len,
  add_start_index = True,
)

prompt_template_no_history = """
You are a chat assistant. Take help of the following pieces of retrieved context in triple backticks to answer the question. If you don't know the answer, just say that you don't know.

```Context: {context} ```

```Question: {query} ```

Answer:
"""

# Get chat history
def get_history(id):
    history = RedisChatMessageHistory(
      session_id=id,
      url=redis_conn,
      key_prefix='ktr:',
      ttl=TTL
    ).messages
    return { "history": list(map(lambda x: { "content": x.content, "type": x.type }, history)) }

# Store file as vector embeddings in ChromaDB
def store_file(data, id, ext):
    if ext == "txt":
      text = data
    elif ext == "pdf":
      text = get_text(id, data)

    if (len(text) == 0):
        return {}

    chunks = text_splitter.create_documents([text])
    collection = client.get_or_create_collection(name=id, embedding_function=openai_ef)

    doc_id = str(uuid4())
    documents = [x.page_content for x in chunks]
    metadata = [x.metadata for x in chunks]
    ids = [f"{doc_id}:{idx}" for idx in range(len(chunks))]
    collection.add(
      documents=documents,
      metadatas=metadata,
      ids=ids
    )
    return { "doc_id": doc_id }

# Query on the knowledge base using context and previous chat history
def knowledge_transfer_query(query, id, retain_history=False):
    if retain_history:
      history = RedisChatMessageHistory(
          session_id=id,
          url=redis_conn,
          key_prefix='ktr:',
          ttl=TTL
      )
      prompt = """
            System : You are a chat assistant. Take help of the following pieces of retrieved context and previous conversation history in triple backticks to answer the question. If you don't know the answer, just say that you don't know.
            Highlight important points and try to structure your answer in a way that is easy to read. Also for technical tasks or terms, link the task or term to a relevant resource (from majorly youtube or if required other relevant resources from web).
            Every answer must have 2-3 links to relevant resources. Links must be in markdown format within the answer text no as a separate list of links.

            ```Context: {context} ```

            """+ f"""```Chat History: {history} ```""" + """
      
            ```Question: {query} ```

            Output Format : Mardown Only
             """
    else:
      prompt = prompt_template_no_history

    output = context_chat(id=id, prompt=prompt, query=query)
    if retain_history:
      history.add_user_message(query)
      history.add_ai_message(output)
    return { "response": output }




   
   

   
   
