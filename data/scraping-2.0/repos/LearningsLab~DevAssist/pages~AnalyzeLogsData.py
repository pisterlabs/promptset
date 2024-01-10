import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import openai
import requests
import json
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from services.GetEnvironmentVariables import GetEnvVariables
from services.TextChunkSplitterService import TextChunkSplitterService
from services.CreateEmbeddingService import CreateEmbeddingService
import re
# get env variables 
env_vars = GetEnvVariables()
OPENAI_API_KEY = env_vars.get_env_variable('openai_key')
openai.api_key=OPENAI_API_KEY

st.set_page_config(page_title="Analyze Logs Data", page_icon="📈")

tab1, tab2, tab3 = st.tabs(["Cloudwatch", "OpenSearch", "ElasticSearch"])

with tab1:
   import boto3
   # Configure AWS credentials and region
   aws_access_key_id = env_vars.get_env_variable('aws_access_key_id')
   aws_secret_access_key = env_vars.get_env_variable('aws_secret_access_key')
   region_name = "ap-south-1"  # Replace with your desired region
   log_group_name = "/aws/lambda/prod-fa-payment-init-new"
   # Initialize the CloudWatch Logs client
   client = boto3.client("logs", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)

   from datetime import datetime, timedelta

   # Set the page size (number of log events to fetch in each API call)
   page_size = 10
   counter=0

   # Fetch log events from all log streams within the log group
   log_events = []

   #Calculate the timestamp range for the last 24 hours
   end_time = datetime.utcnow()
   start_time = end_time - timedelta(hours=24)

   # Convert timestamps to milliseconds
   start_time_ms = int(start_time.timestamp() * 1000)
   end_time_ms = int(end_time.timestamp() * 1000)

   # Get a list of all log streams within the log group
   response = client.describe_log_streams(logGroupName=log_group_name)
   log_streams = response["logStreams"]
   log_stream_name = "2023/05/28/[7]d8a03086c3424b58aa966375a8c19841"

   # Fetch log events from all log streams within the log group
   log_events = []
   # for log_stream in log_streams:
   #     log_stream_name = log_stream["logStreamName"]
   #     print
      # Retrieve log events for the log stream within the desired timestamp range
   next_token = None
   while True:
      params = {
         "logGroupName": log_group_name,
         "logStreamName": log_stream_name,
         "startFromHead": True,
      }
      if next_token:
         params["nextToken"] = next_token
      response = client.get_log_events(**params)
      events = response["events"]
      #print('events',events,response)
      log_events.extend(events)
      next_token = response.get("nextToken")
      if not next_token or counter >= 10:
         break
   

   # Concatenate the strings
   concatenated_text = " "
   for events in log_events:
      concatenated_text += events['message']
      #st.write(concatenated_text)
   
   if(concatenated_text!=" "):
      # split into chunks
      splitters = TextChunkSplitterService('/n',250,200,len)
      chunks = splitters.split_text(concatenated_text)
      st.write(chunks,"chunks")
      
      # create embeddings
      embeddingsIni = CreateEmbeddingService()
      embeddings = embeddingsIni.create_embeddings('OpenEmbeddings')
      
      knowledge_base = FAISS.from_texts(chunks, embeddings)

      user_question = st.text_input("Ask a question to your logs data:")
      if user_question:
         docs = knowledge_base.similarity_search(user_question)
         llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
         chain = load_qa_chain(llm, chain_type="stuff")
         with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            st.write(cb)
      
         st.write("openai res: "+response)
   else :
      st.write("No data available")

with tab2:
   st.header("OpenSearch logs")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("ElasticSearch logs")
  
   def get_elasticsearch_data(index_name, query):
      # Elasticsearch endpoint
      #url = f"http://vpc-similar-product-pwanh5zsjbcnorrre77cazmfce.ap-south-1.es.amazonaws.com/{index_name}/_search"
      url =f"https://search-logs-71-65vca3dfvyuy76y5fcqu53tw6q.ap-south-1.es.amazonaws.com/{index_name}/_search"
      # Request headers
      headers = {
         "Content-Type": "application/json"
      }
      
      # Request body
      body = {
               "query": {
                  "match_all": {}
               },
               "size": 25
            }
      
      # Send the request
      response = requests.get(url, headers=headers, json=body)
      st.write(response)
      # Check the response status
      if response.status_code == 200:
         # Parse and process the response data
         response_data = response.json()
         #st.write(response_data)
         # Handle the retrieved data according to your needs
         return response_data['hits']['hits']
      else:
         # Handle the error case
         st.write(f"Error: {response.status_code}")
         return None
      
   index_name = "logstash-apis"
   query = "Anniversary"

   result = get_elasticsearch_data(index_name, query)
   #st.write(result)

   # Convert JSON data to string
   #json_string = json.dumps(result)
   json_string = ''
   pattern = r"\[.*?\] Production\.INFO: "

   for key in result:
      json_string += re.sub(pattern, "", key['_source']['message'])+"separator "
   
   # print context
   #st.write('context', json_string)
   # split into chunks
   splitters = TextChunkSplitterService('separator',2500,200,len)
   chunks = splitters.split_text(json_string)
   st.write("chunks", chunks)

   #exit()
   # create embeddings
   embeddingsIni = CreateEmbeddingService()
   embeddings = embeddingsIni.create_embeddings('OpenEmbeddings')
   knowledge_base = FAISS.from_texts(chunks, embeddings)
  
   user_question = st.text_input("Ask a question to your Elastic data:")
   if user_question:
      docs = knowledge_base.similarity_search(user_question)
      llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
      chain = load_qa_chain(llm, chain_type="stuff")
      with get_openai_callback() as cb:
         response = chain.run(input_documents=docs, question=user_question)
         st.write(cb)
     
      st.write("openai res: "+response)
   # Process the result


