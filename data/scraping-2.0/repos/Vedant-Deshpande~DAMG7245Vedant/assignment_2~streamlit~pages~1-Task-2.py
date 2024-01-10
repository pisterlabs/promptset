import os
import streamlit as st
from google.cloud import storage
import openai
from google.cloud.sql.connector import Connector
import sqlalchemy

import streamlit as st
import pinecone
from tqdm.auto import tqdm
import openai
from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain



# Get OpenAI API key, Pinecone API key, environment and index, and the source document input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", type="password")
    pinecone_api_key = st.text_input("Pinecone API key", type="password")
    pinecone_environment = st.text_input("Pinecone environment")
    pinecone_index_name = st.text_input("Pinecone index name")

if pinecone_api_key and pinecone_environment and pinecone_index_name:
            # Initialize Pinecone connection
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

            try:
                # Create Pinecone index
                if pinecone_index_name not in pinecone.list_indexes():
                    # Create Pinecone index
                    st.write("Creating Pinecone index. This may take a few minutes...")
                    pinecone.create_index(
                        name=pinecone_index_name,
                        dimension=1536,
                        metric='cosine'
                    )
                    st.write("Pinecone index created.")
                    # Reload page
                    st.experimental_rerun()
            except Exception as e:
                st.write(f"Error while creating Pinecone index: {e}")
            

# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/servicekey.json'

# Set the project ID and bucket name
project_id = 'virtual-sylph-384316'
bucket_name = 'damg7245-assignment-7007'

# Initialize the Google Cloud Storage client
storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)

# Get list of files from GCS
files = [blob.name for blob in bucket.list_blobs()]

# Extract company names and years from files
companies = []
company_years = {}
for file in files:
    company, year = file.split(':')
    companies.append(company)
    if company not in company_years:
        company_years[company] = []
    company_years[company].append(year)

# Remove duplicates
companies = list(set(companies))

#page title - earnings statement
st.markdown("<h1 style='text-align: center; color: white;'>Earnings Statement</h1>", unsafe_allow_html=True)

# User selection
st.markdown("<h2 style='font-weight: bold;'>Select a company</h2>", unsafe_allow_html=True)
company = st.selectbox("Company", companies)

st.markdown("<h2 style='font-weight: bold;'>Select a year</h2>", unsafe_allow_html=True)
year = st.selectbox("Year", company_years[company])

# Get data from GCS
file = f"{company}:{year}"
blob = bucket.blob(file)
data = blob.download_as_text()

connector = Connector()

def getconn():
    conn = connector.connect(
        instance_connection_string="virtual-sylph-384316:us-west1:app",
        driver="pg8000",
        user="postgres",
        password="J1ag[@%$#1.@9k^^",
        db="postgres",
    )
    return conn

pool = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn).connect()

# User choice to view metadata
view_metadata = st.checkbox("View metadata")

if view_metadata:
    # Query Cloud SQL database to retrieve metadata
    import pandas as pd

    # Query database to retrieve metadata
    query = f'SELECT * FROM "Companies_metadata" WHERE "company_name"=\'{company}\' AND EXTRACT(YEAR FROM "date")={year}'
    metadata_df = pd.read_sql(query, pool)

    # Display metadata
    st.write(metadata_df)


if openai_api_key:
    # Truncate data to a maximum of 2048 tokens
    max_tokens = 2048
    data_tokens = data.split()
    if len(data_tokens) > max_tokens:
        data = ' '.join(data_tokens[:max_tokens])

    # Compute summaries using OpenAI
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"   \n\n{data}",
        temperature=0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    summary = response['choices'][0]['text']
    # Truncate summary at the last occurrence of a sentence-ending punctuation mark
    end_punctuation = [".", "!", "?"]
    if summary[-1] not in end_punctuation:
        for i in range(len(summary)-1, -1, -1):
            if summary[i] in end_punctuation:
                summary = summary[:i+1]
                break

    # Split file into company and year
    company, year = file.split(':')

    # Display summary
    st.markdown(f"<h2 style='font-weight: bold;'>Summary of {company} for the year {year}</h2>", unsafe_allow_html=True)
    #display just summary
    st.write(summary)

    # User selection
st.markdown("<h2 style='font-weight: bold;'>Do you want to view the full transcript?</h2>", unsafe_allow_html=True)
    #default should be 'No' for below radio button
view_full_transcript=st.radio("", ("Yes", "No"), index=1)

if view_full_transcript == "Yes":
        # Display full transcript
        st.markdown(f"<h2 style='font-weight: bold;'>Full transcript of {company} for the year {year}</h2> {data}", unsafe_allow_html=True)

        # Set your OpenAI API key
        openai.api_key = openai_api_key

        # Split data into words
        words = data.split()

        # Chunk data into chunks of 500 words
        chunks = [words[i:i + 500] for i in range(0, len(words), 500)]

            # Connect to Pinecone index
        index = pinecone.Index(index_name=pinecone_index_name)

            # Compute embeddings for each chunk and store in Pinecone in batches
        batch_size = 128
        for i in tqdm(range(0, len(chunks), batch_size)):
                # Find end of batch
                i_end = min(i + batch_size, len(chunks))

                # Create IDs batch
                ids = [f"{company}-{year}-chunk-{j}" for j in range(i, i_end)]

                # Compute OpenAI embeddings for chunk
                openai_embeddings = []
                for chunk in chunks[i:i_end]:
                    response = openai.Embedding.create(
                        input=' '.join(chunk),
                        model="text-embedding-ada-002"
                    )
                    openai_embedding = response['data'][0]['embedding']
                    openai_embeddings.append(openai_embedding)

                # Create records list for upsert
                records = zip(ids, openai_embeddings)
                
                

                if pinecone_index_name in pinecone.list_indexes():
                # Upsert to Pinecone
                    try:
                        # Upsert vectors into Pinecone index
                        index.upsert(vectors=records)
                        st.write(f"Upserted batch {i} to {i_end} successfully")
                    except Exception as e:
                        st.write(f"Error while upserting vectors into Pinecone index: {e}")
                

        # Display query text input
        st.markdown("<h2 style='font-weight: bold;'>Ask a question about the transcript</h2>", unsafe_allow_html=True)
        query_text = st.text_input("Query text")

        if query_text:
            # Compute OpenAI embedding for query text
            response = openai.Embedding.create(
                input=query_text,
                model="text-embedding-ada-002"
            )
            query_embedding = response['data'][0]['embedding']

            # Query Pinecone index using query embedding
            results = index.query(queries=[query_embedding], top_k=1)

            # Display query embedding results in Streamlit app
            #st.write(f"embedding results: {results}")

            # Display query results in Streamlit app in bold
            st.markdown(f"<h2 style='font-weight: bold;'>Result</h2>", unsafe_allow_html=True)
            #st.write(f"Query results:")
            for result in results['results'][0]['matches']:
                # Extract ID and score of matching embedding
                match_id = result['id']
                match_score = result['score']

                # Parse ID to extract chunk index
                match_id_parts = match_id.split('-')
                chunk_index = int(match_id_parts[-1])

                # Retrieve text data corresponding to matching embedding
                match_text = ' '.join(chunks[chunk_index])

                # Display matching text data and score in Streamlit app
                st.write(f"- (score: {match_score:.2f}) {match_text}")

# Prompt user to exit
exit = st.button("Exit")

if exit:
    # Delete Pinecone index
    pinecone.delete_index(pinecone_index_name)
    st.write("Pinecone index deleted")
            
elif view_full_transcript == "No":
        # Display thank you message
        st.write("Thank you")