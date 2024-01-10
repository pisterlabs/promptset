from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from astrapy.db import AstraDBCollection

import openai, os, uuid, requests
import pandas as pd

from traceloop.sdk import Traceloop
from traceloop.sdk.tracing import tracing as Tracer
from traceloop.sdk.decorators import workflow, task, agent
from dotenv import load_dotenv, find_dotenv

import streamlit as st
from langchain.llms import OpenAI

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
ASTRA_DB_APPLICATION_TOKEN=os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_API_ENDPOINT=os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_COLLECTION=os.getenv('ASTRA_COLLECTION')

openai.api_key = os.getenv('OPENAI_API_KEY')
model_id = "text-embedding-ada-002"

k=os.getenv('LIMIT_TOP_K')

llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.1)

@st.cache_resource()
@task(name="Establish Astra DB Connection and get Collection")
def create_connection():
    #Establish Connectivity and get the collection
    st.write(":hourglass: Establishing AstraDB Connection...")
    collection = AstraDBCollection(
        collection_name=ASTRA_COLLECTION, token=ASTRA_DB_APPLICATION_TOKEN, api_endpoint=ASTRA_DB_API_ENDPOINT
    )
    return collection

@task(name="Embed Input Query")
def embed_query(customer_input):
    # Create embedding based on same model
    st.write(":hourglass: Using OpenAI to Create Embeddings for Input Query...")
    prompt = f"Suggest a response to the customer inquiry: {customer_input}? Respond in format that would be suitable for searching a database of professional bike reviews."
    embedding = openai.embeddings.create(input=prompt, model=model_id).data[0].embedding
    return embedding

@task(name="Build top k simple query")
def build_simple_query(customer_input, k):
    st.write(":hourglass: Building Simple Database Query...")
    params = {}
    params['embedding'] = embed_query(customer_input)
    params['k'] = k
    
    return params

@task(name="Build top k hybrid query")
def build_hybrid_query(customer_input, filter, k):
    st.write(":hourglass: Building Hybrid Search Query...")
    params = {}
    params['embedding'] = embed_query(customer_input)
    params['k'] = k
    params['filter'] = filter
    return params

@task(name="Perform ANN search on Astra DB")
def query_astra_db(collection, params):
    st.write(":hourglass: Retrieving results from Astra DB...")
    if 'filter' in params:
        results = collection.vector_find(
            vector=params['embedding'],
            limit=params['k'],
            filter={"type": params['filter']},
            fields=["type", "brand", "model", "price", "description", "image"],
        )
    else:
        results = collection.vector_find(
            vector=params['embedding'],
            limit=params['k'],
            fields=["type", "brand", "model", "price", "description", "image"],
        )
    
    bikes_results = pd.DataFrame(results)
    return bikes_results

@task(name="Build table with Bike Reco Results")
def create_display_table(bikes_results):
    st.dataframe(
        bikes_results,
        column_config={
            "brand": "Brand Name",
            "model": "Model",
            "description": "Desc",
            "price": "Price (in USD)",
            "image": st.column_config.ImageColumn(
                "Bike Image", help="Recommended Bike Image", width = "small"
                ),
            "type": "Bike Type",
        },
        column_order=("image", "brand", "model", "type", "price", "description"),
        hide_index=True,
        )

@task(name="Ask ChatGPT to Generate a Professional Recommendation")
def create_display_cgpt_response(bikes_results, customer_input):
    bike_desc_prompts=[]

    for index, row in bikes_results.iterrows():
        reco_prompt = f"You are an experienced bike rider who is working at a bike retail shop Bike Mart and your job is to provide Bike Recommendations to customers. For a {customer_input}, why would you suggest {row['model']} {row['brand']}, described as {row['description']}?"
        bike_desc_prompts.append(reco_prompt)

    st.write(":bicyclist: Generating Bike recommendations Using ChatGPT :bicyclist:")

    recommendations = llm.generate(bike_desc_prompts)

    st.write(":bicyclist: Here are some Bike recommendations:")

    desc_list=''
    for i, generation in enumerate(recommendations.generations):
        description = generation[0].text.strip('\n')
        desc_list += f"- {description}\n"
    
    st.write(desc_list)

@workflow(name="Bike Recommendation Demo UI")
def execute_demo_ui():
    ##################################
    st.title('🚲 Bike Recommendation Agent')
    st.write("These Bike Recommendations are based on sample dataset.")
    with st.expander('**Scenario Details**'):
        st.write("""
    The Bike Review and Descriptions have been embedded using [OpenAI's Text Embedding API](https://beta.openai.com/docs/api-reference/text-embeddings),
    and stored in DataStax Astra.

    Based on the bike recommendation asked, this demo will:

    1. Search the vector database for the best bike matches, using these reviews and description.
    2. User have the ability to further instruct on what type of bike.
    3. ChatGPT completion API is used to clean the description as per context.

    """)

    query = st.text_input('Please Enter your Bike Question:')
    filter = st.text_input("Please Enter Bike Type: (e.g. Kids Bike or eBikes) ***Optional***")

    if st.button('Ask Me! :bicyclist:'):
        if query:
            if filter:
                collection = create_connection()
                db_query = build_hybrid_query(query, filter, k)
                bikes_results = query_astra_db(collection, db_query)
                if bikes_results.empty:
                    st.error("No Response received")                    
                else:
                    create_display_table(bikes_results)
                    create_display_cgpt_response(bikes_results, query)
            else:
                collection = create_connection()
                db_query = build_simple_query(query, k)
                bikes_results = query_astra_db(collection, db_query)
                if bikes_results.empty:
                    st.error("No Response received")                    
                else:
                    create_display_table(bikes_results)
                    create_display_cgpt_response(bikes_results, query)
        else:
            st.error("Please provide a question to start!")

#call main method
execute_demo_ui()