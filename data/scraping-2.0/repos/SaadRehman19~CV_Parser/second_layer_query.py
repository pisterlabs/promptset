import os
import shutil
import openai
import streamlit as st
import base64
from flask import send_file
import mysql.connector
import time

# host = '127.0.0.1'
# user = 'root'
# password = 'root'
# port = '42333'
# database = 'CV'

# mydb=mysql.connector.connect(
#             host=host,
#             user=user,
#             password=password,
#             port=port,
#             database=database
#         )
def second_layer(filename,filecontent,text):

    openai.api_key = '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'

    query=text

    # prompt=f"""
    # You are a HR Manager And Your Task is to read The Resume and query and provide me the relevancy or recommendation between resume and query in percentage in given format.Give the result in specified format only \n

    # Resume: {filecontent}  \n
    # Query: {query} \n
    # Format: x% Similarity
    # """

    # prompt=f"""
    # Identity: You are a Hiring Manager in ABC Company.\n
    # Task: Your Task Is To Read The Resume and Query and Provide the relevancy or resemblance Between Resume and Query in Percentage in Given Format.\n
    # Context: Read The Resume And Query Below And Answer the Above specified Questions.\n
    # Resume: {filecontent} \n
    # Query: {query } \n
    # Format: x% Similarity, Reason:..
    # """

    prompt=f'You Are a Hiring Manager and your task is to read the resume and query and provide me the relevancy(resemblance) between Resume and Query in Percentage Only \n Resume : {filecontent} \n Query: {query} \n Give the output in given format \n Format:x% Similarity,Reason:....."""'
    

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}],
        temperature=0.02
       )
    

    output=response.choices[0].message["content"]

    # print(filename +":",output)
    st.write(f"{filename} : {output}")
    # st.write(f"{filename} \nSimilarity : {similarity} \nReason: {reason}")
    file_path = f'/home/gaditek/CV_To_WordEmbedding/CV_File/{filename}' 

     # Read the PDF file
    with open(file_path, "rb") as file:
        pdf_content = file.read()

    b64_pdf = base64.b64encode(pdf_content).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" style="text-decoration: none; color: #3366FF; font-weight: bold;">Click here to download this Resume</a>'
    st.markdown(href,unsafe_allow_html=True)

   


