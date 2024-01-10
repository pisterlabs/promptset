from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory, SimpleStatement
import openai, os, uuid, time, requests, traceback
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing import tracing as Tracer
from traceloop.sdk.decorators import workflow, task, agent
from dotenv import load_dotenv, find_dotenv
import pandas as pd
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
ASTRA_DB_SECURE_BUNDLE_PATH=os.getenv('ASTRA_SECUREBUNDLE_PATH')
ASTRA_DB_APPLICATION_TOKEN=os.getenv('ASTRA_DB_TOKEN')
ASTRA_DB_KEYSPACE=os.getenv('ASTRA_KEYSPACE')
k=os.getenv('LIMIT_TOP_K')
openai.api_key = os.getenv('OPENAI_API_KEY')
model_id = "text-embedding-ada-002"
llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.1)

@st.cache_resource()
@task(name="Create Cassandra Connection")
def create_connection():
    #Establish Connectivity
    st.write(":hourglass: Establishing AstraDB Connection...")
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
    keyspace = ASTRA_DB_KEYSPACE
    return session, keyspace

@task(name="Embed Input Query")
def embed_query(customer_input):
    # Create embedding based on same model
    st.write(":hourglass: Using OpenAI to Create Embeddings for Input Query...")
    prompt = f"Please suggest {customer_input}? Please respond in format that would be suitable for searching a database of professional bike reviews."
    embedding = openai.Embedding.create(input=prompt, model=model_id)['data'][0]['embedding']
    return embedding

@task(name="Build top k simple query")
def build_simple_query(customer_input, keyspace, k):
    st.write(":hourglass: Building Simple Database Query...")
    embedding = embed_query(customer_input)
    query = SimpleStatement(
    f"""
    SELECT *
    FROM {keyspace}.bikes
    ORDER BY description_embedding ANN OF {embedding} LIMIT {k};
    """
    )
    return query

@task(name="Build top k hybrid query")
def build_hybrid_query(customer_input, keyspace, filter, k):
    st.write(":hourglass: Building Hybrid Search Query...")
    embedding = embed_query(customer_input)
    hybrid_query = SimpleStatement(
    f"""
    SELECT *
    FROM {keyspace}.bikes
    WHERE type : '{filter}'
    ORDER BY description_embedding ANN OF {embedding} LIMIT {k};
    """
    )
    return hybrid_query

@task(name="Perform ANN search on Astra DB")
def query_astra_db(session, query):
    st.write(":hourglass: Retrieving results from Astra DB...")
    results = session.execute(query)
    top_results = results._current_rows
    bikes_results = pd.DataFrame(top_results)
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
                session, keyspace = create_connection()
                db_query = build_hybrid_query(query, keyspace, filter, k)
                bikes_results = query_astra_db(session, db_query)
                if bikes_results.empty:
                    st.error("No Response received")                    
                else:
                    create_display_table(bikes_results)
                    create_display_cgpt_response(bikes_results, query)
            else:
                session, keyspace = create_connection()
                db_query = build_simple_query(query, keyspace, k)
                bikes_results = query_astra_db(session, db_query)
                if bikes_results.empty:
                    st.error("No Response received")                    
                else:
                    create_display_table(bikes_results)
                    create_display_cgpt_response(bikes_results, query)
        else:
            st.error("Please provide a question to start!")

#call main method
execute_demo_ui()