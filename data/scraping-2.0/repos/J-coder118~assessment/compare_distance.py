import chromadb
from chromadb.utils import embedding_functions
import openai
import csv
import pandas as pd
import streamlit as st
import mysql.connector

# connection = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="chroma"
# )

# if connection.is_connected():
#     print("Connected to MySQL database")

# # Creating a cursor to interact with the database
# cursor = connection.cursor()

# # Executing a SELECT query to retrieve data from a table
# query = "SELECT * FROM chromadbtbl"
# cursor.execute(query)

# # Fetching all the rows from the result set
# rows = cursor.fetchall()

# # Getting column names
# column_names = [description[0] for description in cursor.description]

# # Convert data to array for each column
# column_data = [[] for _ in range(len(column_names))]

# for row in rows:
#     for idx, value in enumerate(row):
#         column_data[idx].append(value)


# Unitid = column_data[0]
# CollegeName = column_data[1]
# City = column_data[2]
# State = column_data[3]
# TitleOfProgram = column_data[4]
# ProgramCode = column_data[5]
# Tags = column_data[6]
# ReferringUrl = column_data[7]
# Num = column_data[8]
# Content = column_data[9]

# cursor.close()

# documents = []
# documents.append([cn + ", " + city + ", " + state + ", " + top + ", " + pc + ", " + tags + ", " + ru  + ", " + str(num) + ", " + content  for cn, city, state, top, pc, tags, ru, num, content in zip(CollegeName, City, State, TitleOfProgram, ProgramCode, Tags, ReferringUrl, Num, Content)])

# print("----------------- Get data from database  -------------------------")
# # print("documents", Content)

# #
# ###### insert by csv
# # df = pd.read_csv('doc/merged_file.csv')

# # df = df.dropna()

# # All = ['"' + str(value) + '",' for value in df['All'].values]
# ######
# print("----------------- create vector database  -------------------------")

documents = []

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = ""

if client:
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # collection = client.get_or_create_collection(
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
        # metadata={"hnsw:space": "cosine"},
    )
    print("----------------------  use the existing collection ------------------")

else :
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # collection = client.get_or_create_collection(
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        documents=documents,
        ids=[f"id{i}" for i in range(len(documents))],
        # metadatas=[{"CollegeName": cn, "City": city, "State": state, "TitleOfProgram": top, "ProgramCode": pc,"Tags": tags, "ReferringUrl": ru } for cn, city, state, top, pc, tags, ru in zip(CollegeName, City, State, TitleOfProgram, ProgramCode, Tags, ReferringUrl)
        # ])
    )

    print("----------------------  creating the collection ------------------")





openai.api_key = ""

context = """
You are question answer system service to answer to make understand easily.
Use following context to answer in more detail to user question.
If you can't find answer, just summarize the context.
context: {}

"""

st.title("chat")
st.caption("Please ask what you want.")

question = st.text_input("Question")
# question = input("question:")
if st.button("Get Answer"):
    query_result = collection.query(
        query_texts=[f"{question}"],
        n_results=3,
    )
    # print(query_result["distances"])
    # print(query_result["ids"][0])
    # print(query_result["documents"][0])
    # print(query_result["documents"][0])
    contents = ",".join(query_result["documents"][0])
    # print("contents", contents)
    good_answer = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
         {"role": "system", "content": context.format(contents)},
         {"role": "user", "content": question},
     ],
    temperature=0,
    n=1,
    )

    # print(query_result.keys())
    # print(query_result["metadatas"])

    # st.write("Content:", query_result["metadatas"][0][0]["CollegeName"], query_result["documents"])
    # st.write("Answer:", good_answer["choices"][0]["message"]["content"])
    # print("answer:", good_answer["choices"][0]["message"]["content"])
##  assessment
    print(query_result["distances"])