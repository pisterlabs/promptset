print('starting vector indexer...')

import ast
import os, json, sys
import langchain
from contextlib import redirect_stdout
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from google.cloud import pubsub
from google.oauth2 import service_account

print('langchain: {version}'.format(version=langchain.__version__))

PROJECT_ID = 'broyal-llama-demo'
if os.getenv("DEBUG"):
    f = open("../.pubsub-svc/pubsub-svc.json", "r")
    secret = json.loads(f.read())
else:
    f = open("/etc/secret-volume/pubsub-svc.json", "r")
    secret = json.loads(f.read())

credentials = service_account.Credentials.from_service_account_info(secret)
subscriber = pubsub.SubscriberClient(credentials=credentials)
subscription_path = subscriber.subscription_path(PROJECT_ID, os.getenv("PUBSUB_SUBSCRIPTION",'kubernetes_concepts_subscription'))
print('connecting to pubsub subscription: {subscription}'.format(subscription=subscription_path))

def process_data(urls):
    print('start: processing data')
    loader = WebBaseLoader(urls)
    documents = loader.load()

    # Chunk all the kubernetes concept documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    print("%s chunks in %s pages" % (len(docs), len(documents)))
    return docs


# Load sentence transformer embeddings
def load_embeddings():
    print('start: loading embeddings')
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device":"cpu"} # use {"device":"cuda"} for distributed embeddings
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

def test_connection():
  import socket
  postgres_host=socket.gethostbyname(os.environ.get("PGVECTOR_HOST", "localhost"))
  print('postgres host: {host}'.format(host=postgres_host))

def get_connection_string():
    # print('start: getting connections')
    return PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
        user=os.environ.get("PGVECTOR_USER", "postgres"),
        password=os.environ.get("PGVECTOR_PASSWORD", "secretpassword"),
    )

test_connection()
embeddings = load_embeddings()
print('connecting to pgVector store')

db = PGVector.from_existing_index(
    collection_name=os.getenv("COLLECTION_NAME","kubernetes_concepts"),
    connection_string=get_connection_string(),
    embedding=embeddings,
)

while True:
  response = subscriber.pull(
    request={
      "subscription": subscription_path,
      "max_messages": 5,
    }
  )

  if not response.received_messages:
    print('❌ no documents in pub/sub topic')
    break
  
  urls = []
  for msg in response.received_messages:
    print(msg.message)
    url = msg.message.data.decode("utf-8")
    urls.append(url)
    # message_data = ast.literal_eval(msg.message.data.decode('utf-8'))
    msg
  print("starting index of {urls}".format(urls=urls))
  docs = process_data(urls)

  collection_name=os.getenv("COLLECTION_NAME","kubernetes_concepts")
  print('connectingn to vectordb. adding documents to {collection_name}'.format(collection_name=collection_name))

  ack_ids = [msg.ack_id for msg in response.received_messages]
  subscriber.acknowledge(
    request={
      "subscription": subscription_path,
      "ack_ids": ack_ids,
    }
  )

print('🏁 No more documents left in the queue. Shutting down...')
sys.exit() # adding to force exit code 0 when successfully shutting down