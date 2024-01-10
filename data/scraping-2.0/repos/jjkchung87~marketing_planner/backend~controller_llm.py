from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import os

openai_api_key = os.environ.get('OPENAI_API_KEY')
data = pd.read_csv('campaign_seed_data.csv')
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0, openai_api_key=openai_api_key)
agent = create_pandas_dataframe_agent(chat, data, verbose=True)

def get_chat_response(prompt):
    """
    Sends a POST request to the chat completion API with a prompt.
    :param prompt: The prompt to be sent to the API.
    """
    try:
        response = agent(prompt)
        return response
    except:
        return "Sorry, I don't understand that. Please try again."
    


# **********************************************************************************************************************
# function to ingest a file to PrivateGPT API
# **********************************************************************************************************************

# import requests
# from requests_toolbelt.multipart.encoder import MultipartEncoder


# def post_file_to_ingest_api(file_path):
#     """
#     Sends a POST request to the ingest API with a file.
    
#     :param file_path: The path to the file to be uploaded.
#     """
#     url = "http://localhost:8000/v1/ingest/file"  # Ensure 'http://' is included in the URL.
    
#     # Prepare the multipart/form-data payload

#     # file type can be plain text, pdf, docx, pptx, xlsx, csv, json, xml, html, or markdown

#     multipart_data = MultipartEncoder(
#         fields={'file': ('filename', open(file_path, 'rb'), 'text/csv')}
#     )
    
#     # Set the headers
#     headers = {
#         'Content-Type': multipart_data.content_type, 
#         'Accept': 'application/json'
#     }
    
#     # Send the request
#     response = requests.post(url, headers=headers, data=multipart_data)
    
#     # Handle the response
#     if response.status_code == 200:
#         print("File uploaded successfully.")
#         return response.json()  # or `response.content` depending on the expected response
#     else:
#         print(f"Failed to upload file. Status code: {response.status_code}")
#         return None

# # **********************************************************************************************************************
# # function for chat completions
# # **********************************************************************************************************************

# def post_chat_completion_api(prompt):
#     """
#     Sends a POST request to the chat completion API with a prompt.
#     :param prompt: The prompt to be sent to the API.
#     """
#     url = "http://localhost:8000/v1/chat/completions"  # Ensure 'http://' is included in the URL.
        
#     # Set the headers
#     headers = {
#         'Content-Type': 'application/json', 
#         'Accept': 'application/json'
#     }
    
#     data = {
#         "context_filter": {
#             "docs_ids": [
#             "9854b485-b966-43c5-91ee-e052f67f87b3"
#             ]
#         },
#         "include_sources": True,
#         "messages": [
#             {
#             "content": "You are a helpful data analyst.",
#             "role": "system"
#             },
#             {
#             "content": f"{prompt}",
#             "role": "user"
#             }
#         ],
#         "stream": False,
#         "use_context": False
#         }
#     # Send the request
#     response = requests.post(url, headers=headers, data=data)
    
#     # Handle the response
#     if response.status_code == 200:
#         print("Chat completion successful.")
#         return response.choices[0].content
#     else:
#         print(f"Failed to complete chat. Status code: {response.status_code}")
#         return None
    




