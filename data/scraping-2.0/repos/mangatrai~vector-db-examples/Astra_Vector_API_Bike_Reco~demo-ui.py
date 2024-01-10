import os, json
from dotenv import load_dotenv, find_dotenv
from astrapy.db import AstraDB
from openai import OpenAI
import streamlit as st
from langchain.llms import OpenAI as OpenAILC

# Load the .env file
if not load_dotenv(find_dotenv(),override=True):
    raise Exception("Couldn't load .env file")

#declare constant
ASTRA_DB_API_ENDPOINT=os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN=os.getenv('ASTRA_DB_TOKEN')
ASTRA_NAMESPACE=os.getenv('ASTRA_NAMESPACE')
ASTRA_COLLECTION=os.getenv('ASTRA_COLLECTION')
k=os.getenv('LIMIT_TOP_K')
model_id = "text-embedding-ada-002"

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

llm = OpenAILC(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0.1)

@st.cache_resource()
def create_collection(astra_db_api_endpoint,astra_db_token, astra_namespace,astra_create_collection):
    #Establish Connectivity
    astra_db = AstraDB(
    api_endpoint=astra_db_api_endpoint,
    token=astra_db_token,
    namespace=astra_namespace
    )
    astra_collection_obj = astra_db.collection(astra_create_collection)
    return astra_collection_obj

def embed_query(customer_input):
    # Create embedding based on same model
    embedding = client.embeddings.create(input= customer_input, model=model_id).data[0].embedding
    return embedding

def execute_simple_query(customer_input, astra_collection_obj, k):
    embedding = embed_query(customer_input)
    bike_results = astra_collection_obj.vector_find(embedding, limit=k, fields=["model", "brand", "price", "type", "image", "description"])
    return bike_results

def execute_hybrid_query(customer_input, astra_collection_obj, metadata_filter, k):
    embedding = embed_query(customer_input)
    bike_results = astra_collection_obj.vector_find(embedding,limit=k,filter={"type": metadata_filter}, fields=["model", "brand", "price", "type", "image", "description"])
    return bike_results

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

def create_display_cgpt_response(bikes_results, customer_input):
    bike_desc_prompts=[]

    for row in bikes_results:
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

    """)

    query = st.text_input('Please Enter your Bike Question:')
    filter = st.text_input("Please Enter Bike Type: (e.g. Kids Bike or eBikes) ***Optional***")

    if st.button('Ask Me! :bicyclist:'):
        if query:
            if filter:
                astra_collection_obj = create_collection(ASTRA_DB_API_ENDPOINT,ASTRA_DB_APPLICATION_TOKEN,ASTRA_NAMESPACE,ASTRA_COLLECTION)
                bikes_results = execute_hybrid_query(query, astra_collection_obj, filter, k)
                if len(bikes_results) > 0:
                    create_display_table(bikes_results)
                    create_display_cgpt_response(bikes_results, query)                    
                else:
                    st.error("No Recommendation Generated")
            else:
                astra_collection_obj = create_collection(ASTRA_DB_API_ENDPOINT,ASTRA_DB_APPLICATION_TOKEN,ASTRA_NAMESPACE,ASTRA_COLLECTION)
                bikes_results = execute_simple_query(query, astra_collection_obj, k)
                if len(bikes_results) > 0:
                    create_display_table(bikes_results)
                    create_display_cgpt_response(bikes_results, query)                
                else:
                    st.error("No Recommendation Generated")
        else:
            st.error("Please provide a question to start!")

#call main method
execute_demo_ui()