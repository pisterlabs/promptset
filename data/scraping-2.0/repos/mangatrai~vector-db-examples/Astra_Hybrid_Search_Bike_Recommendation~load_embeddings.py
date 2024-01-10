from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory, SimpleStatement
import openai, os, uuid, time, requests, traceback, math
import pandas as pd
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing import tracing as Tracer
from traceloop.sdk.decorators import workflow, task, agent
from dotenv import load_dotenv, find_dotenv

# Load the .env file
if not load_dotenv(find_dotenv(),override=True):
    raise Exception("Couldn't load .env file")

#Add Telemetry
TRACELOOP_API_KEY=os.getenv('TRACELOOP_API_KEY')
Traceloop.init(app_name="Bike Recommendation App", disable_batch=True)
# Generate a UUID
uuid_obj = str(uuid.uuid4())
Tracer.set_correlation_id(uuid_obj)

#declare constant
ASTRA_DB_SECURE_BUNDLE_PATH=os.getenv('ASTRA_SECUREBUNDLE_PATH')
ASTRA_DB_APPLICATION_TOKEN=os.getenv('ASTRA_DB_TOKEN')
ASTRA_DB_KEYSPACE=os.getenv('ASTRA_KEYSPACE')
openai.api_key = os.getenv('OPENAI_API_KEY')
model_id = "text-embedding-ada-002"

@task(name="Create Cassandra Connection")
def create_connection():
    #Establish Connectivity
    cluster = Cluster(
    cloud={
        "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH,
    },
    auth_provider=PlainTextAuthProvider(
        "token",
        ASTRA_DB_APPLICATION_TOKEN,
    ),
    )
    session = cluster.connect()
    print("DB Session Created")
    keyspace = ASTRA_DB_KEYSPACE
    return session, keyspace

@task(name="Drop Cassandra Objects")
def drop_db_objects(session, keyspace):
    session.execute(f"""DROP INDEX IF EXISTS {keyspace}.descriptions_vector_idx""")
    session.execute(f"""DROP INDEX IF EXISTS {keyspace}.type_idx_analyzer""")
    session.execute(f"""DROP TABLE IF EXISTS {keyspace}.bikes""")
    print("DB Object dropped")

@task(name="Create Cassandra Objects")
def create_db_objects(session, keyspace):
    session.execute(f"""CREATE TABLE IF NOT EXISTS {keyspace}.bikes
                    (model text,
                    brand text,
                    type text,
                    price decimal,
                    image text,
                    description text,
                    description_embedding vector<float, 1536>,
                    PRIMARY KEY (brand,model))""")
    session.execute(f"""CREATE CUSTOM INDEX IF NOT EXISTS descriptions_vector_idx ON {ASTRA_DB_KEYSPACE}.bikes (description_embedding) USING 'org.apache.cassandra.index.sai.StorageAttachedIndex' WITH OPTIONS = {{'similarity_function':'dot_product'}}""")
    session.execute(f"""CREATE CUSTOM INDEX IF NOT EXISTS type_idx_analyzer ON {ASTRA_DB_KEYSPACE}.bikes (type) USING 'org.apache.cassandra.index.sai.StorageAttachedIndex' WITH OPTIONS = {{'index_analyzer': '{{"tokenizer" : {{"name" : "standard"}},"filters" : [{{"name" : "porterstem"}},{{"name" : "lowercase",	"args": {{}}}}]}}'}};""")
    print("DB Objects Created")

@task(name="Download Raw json data file from source")
def load_data_file():
    #load data from sample json file
    url = "https://raw.githubusercontent.com/mangatrai/vector-db-examples/main/Astra_Hybrid_Search_Bike_Recommendation/data/bikes-updated.json"
    response = requests.get(url)
    bikes = response.json()
    print("Bike file loaded")
    bikes = pd.DataFrame(bikes)
    return bikes

@task(name="Create and load Embeddings")    
def create_load_embeddings(bikes, session):
   for id in bikes.index:
      description = bikes['description'][id].replace(',', '\,')
      description = description.replace('"', '\"')
      image = bikes['image'][id]

      try:
         if float(image):
             image ="https://img1.cgtrader.com/items/3587445/dcfbb2669c/large/road-bike-generic-rigged-3d-model-max.jpg"
      except ValueError:
          # do nothing
          pass

      # Create Embedding for each bike row, save them to the database
      full_chunk = bikes['description'][id]
      embedding = openai.Embedding.create(input=full_chunk, model=model_id)['data'][0]['embedding']
      #print("Embeddings Created - Count " + str(id))
      query = SimpleStatement(f"""INSERT INTO bike_rec.bikes(model, brand, price, image, type, description, description_embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)""")
     
      # Create a try-catch block
      try:
         session.execute(query, (bikes['model'][id], bikes['brand'][id], bikes['price'][id], image, bikes['type'][id], description, embedding), trace=True)
         print("Record Inserted: " + str(id))
      except Exception as e:
          # Log the exception
          traceback.print_exc()
          print(e)
          break

@workflow(name="Load Bike Recommendation Data")
def run_loading_data():
    #Create Connection
    session, keyspace = create_connection()
    
    #database operations
    drop_db_objects(session, keyspace)
    create_db_objects(session,keyspace)

    bikes = load_data_file()
    
    #call embedding function and load data
    create_load_embeddings(bikes, session)

run_loading_data()




