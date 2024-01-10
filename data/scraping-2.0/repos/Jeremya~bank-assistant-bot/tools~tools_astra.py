from langchain.tools import BaseTool

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from dotenv import dotenv_values
import openai

import streamlit as st

### parameters #########
config = dotenv_values('.env')
openai.api_key = config['OPENAI_API_KEY']
astra_or_chroma = config['ASTRA_OR_CHROMA']

if astra_or_chroma == "astra":
    SECURE_CONNECT_BUNDLE_PATH = config['SECURE_CONNECT_BUNDLE_PATH']
    ASTRA_CLIENT_ID = config['ASTRA_CLIENT_ID']
    ASTRA_CLIENT_SECRET = config['ASTRA_CLIENT_SECRET']
    ASTRA_KEYSPACE_NAME = config['ASTRA_KEYSPACE_NAME']

    # Open a connection to the Astra database
    cloud_config = {
        'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
    }
    auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_client = cluster.connect()

### Total Revenue Reader Tool #########
class TotalRevenueReaderTool(BaseTool):
    name = "Total Revenue Reader"
    description = "This tool will read the total revenue by client and return it so you can see it."

    def _run(self, client_id):  # add user uuid
        client_id = "1"
        query = f"SELECT total_revenue FROM {ASTRA_KEYSPACE_NAME}.TotalRevenueByClient WHERE client_id = {client_id}"
        rows = astra_client.execute(query)
        for row in rows:
            st.write(row.total_revenue)

        return row.total_revenue

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


### Client Similarity Tool Astra #########
class ClientSimilarityTool(BaseTool):
    name = "Client Similarity Tool"
    description = "This tool is used to search for client information like balance, credit score, has a credit card, " \
                  "gender, surname, location, point earned and satisfaction score. " \
                  "Note this does not contains user names or emails." \
                  "Example query: what is the top 3 client in alabama ranked by credit score?"

    def _run(self, user_question):
        model_id = "text-embedding-ada-002"
        embedding = openai.Embedding.create(input=user_question, model=model_id)['data'][0]['embedding']
        query = f"SELECT client_id, surname, credit_score, location, gender, age, balance, has_credit_card, " \
                f"estimated_salary, satisfaction_score, card_type, point_earned FROM {ASTRA_KEYSPACE_NAME}.ClientById " \
                f"ORDER BY embedding_client ANN OF {embedding} LIMIT 5 "
        rows = astra_client.execute(query)

        client_list = []
        for row in rows:
            client_list.append({f"client id is {row.client_id}, current balance is {row.balance}, client's surname is {row.surname}, age is {row.age}, gender is {row.gender} , card type owned by client is {row.card_type}, credit score is {row.credit_score}, satisfaction score is {row.satisfaction_score}, point earned is {row.point_earned}, this client is located in {row.location}, client has a credit card is {row.has_credit_card}"})
        return client_list

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

# Get Client Information Tool Astra
class GetClientInformationTool(BaseTool):
    name = "Get Client Information"
    description = "This tool will get the client information"

    def _run(self, client_id):
        query = f"SELECT client_id, surname, credit_score, location, gender, age, balance, has_credit_card, " \
                f"estimated_salary, satisfaction_score, card_type, point_earned FROM {ASTRA_KEYSPACE_NAME}.ClientById WHERE client_id = {client_id}"
        rows = astra_client.execute(query)
        client_list = []
        for row in rows:
            client_list.append({f"client id is {row.client_id}, current balance is {row.balance}, client's surname is {row.surname}, age is {row.age}, gender is {row.gender} , card type owned by client is {row.card_type}, credit score is {row.credit_score}, satisfaction score is {row.satisfaction_score}, point earned is {row.point_earned}, this client is located in {row.location}, client has a credit card is {row.has_credit_card}"})
        return client_list

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


