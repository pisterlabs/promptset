import csv
import json

import openai
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from dotenv import dotenv_values
from cassandra.query import SimpleStatement

# Description: This file will load the clients dataset into Astra DB

# parameters #########
config = dotenv_values('../.env')
openai.api_key = config['OPENAI_API_KEY']
SECURE_CONNECT_BUNDLE_PATH = config['SECURE_CONNECT_BUNDLE_PATH']
ASTRA_CLIENT_ID = config['ASTRA_CLIENT_ID']
ASTRA_CLIENT_SECRET = config['ASTRA_CLIENT_SECRET']
ASTRA_KEYSPACE_NAME = config['ASTRA_KEYSPACE_NAME']
model_id = "text-embedding-ada-002"

# Open a connection to the Astra database
cloud_config = {
    'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

# This function will load a CSV and insert values into the Astra database
# Input format:
# RowNumber,CustomerId,Surname,
# CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited,
# Complain,Satisfaction Score,Card Type,Point Earned
#
# Astra table columns:
# client_id, surname, credit_score, location, gender, age, balance, has_credit_card,
# estimated_salary, satisfaction_score, card_type, point_earned, embedding_client

with open('../resources/clients-dataset.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    query = SimpleStatement(f"INSERT INTO {ASTRA_KEYSPACE_NAME}.ClientById (client_id, surname, credit_score, location, gender, age, " \
            "balance, has_credit_card, estimated_salary, satisfaction_score, card_type, point_earned, " \
            "embedding_client) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

    for row in reader:
        # Create a dictionary for the row using headers as keys
        row_dict = dict(zip(headers, row))

        # Insert client information and embedding into chroma
        json_data_row = json.dumps(row_dict)

        # Create embedding for client containing all the columns
        embedding_client = openai.Embedding.create(input=json_data_row, model=model_id)['data'][0]['embedding']

        # Insert values into Astra database
        session.execute(query, (int(row_dict['CustomerId']), row_dict['Surname'], int(row_dict['CreditScore']), row_dict['Geography'], row_dict['Gender'], int(row_dict['Age']), float(row_dict['Balance']), bool(row_dict['HasCrCard']),
                                float(row_dict['EstimatedSalary']), int(row_dict['Satisfaction Score']), row_dict['Card Type'], int(row_dict['Point Earned']), embedding_client))

        print(f"Inserted client {row_dict['CustomerId']} into Astra DB")

# Close the connection to the Astra database
session.shutdown()

## TODO
# failed to bind prepared statement on embedding type
# cassandra.InvalidRequest: Error from server: code=2200 [Invalid query] message="cannot parse '?' as hex bytes"