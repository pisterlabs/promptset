import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask import session
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import (
    SemanticSettings,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField
)

load_dotenv()
# Configure environment variables  
load_dotenv()  
openai.api_type: str = "azure"  
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")  
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")  
model: str = "text-embedding-ada-002"
# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
model: str = "text-embedding-ada-002"
vector_store_address: str = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY") 
index_name: str = "cresendev"


# Initialize gpt-35-turbo and our embedding model
llm = AzureChatOpenAI(deployment_name="gpt-35-turbo-16k", openai_api_version="2023-03-15-preview", openai_api_key = "70cfad14bc804b44a3b2b294e98ddbe6", openai_api_base = "https://cresen-open-ai.openai.azure.com/") # For Chat
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', deployment='text-embedding-ada-002', chunk_size=1, openai_api_key = "70cfad14bc804b44a3b2b294e98ddbe6", openai_api_base = "https://cresen-open-ai.openai.azure.com/") # For Embeddings

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    semantic_configuration_name='config',
        semantic_settings=SemanticSettings(
            default_configuration='config',
            configurations=[
                SemanticConfiguration(
                    name='config',
                    prioritized_fields=PrioritizedFields(
                        title_field=SemanticField(field_name='content'),
                        prioritized_content_fields=[SemanticField(field_name='content')],
                        prioritized_keywords_fields=[SemanticField(field_name='metadata')]
                    ))
            ])
    )


# Set up retrieval chain
general_system_template = r""" 
You are 'CresenPharmaGPT'.
Find all the information from the provided documents.
If the question pertains to Settlement Agreements, Product Whitepapers, or FDA Warning letters, answer with the expertise of a healthcare compliance expert.
If there's a contract document provided, extract the terms and qualifications required for the individual. Based on this, and the healthcare professional's sanction status (if provided), determine if the person is fit for the mentioned contract or not.
If determining a healthcare professional's sanction status, determine based on the latest year available whether the person is currently sanctioned or not. Also mention the latest year based on which the sanction status is decided.
Answer only in English language. Do not confuse drug names with other languages.
Give detailed, lengthy responses if possible.
 ----
{context}
----
"""

general_user_template = "Question:```{question}```"
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,combine_docs_chain_kwargs={'prompt': qa_prompt},
                                           return_source_documents=True)

from flask import Flask, session
from flask_session import Session

app = Flask(__name__)

# Session configuration should be done once
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SECRET_KEY"] = 'POnNOsbdnqxyimIft/oCUMwEowQr9kAW'
Session(app)

CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

#enter credentials
account_name = 'cresendrugdata'
account_key = 'Hy4tA8wAO+6pOHFGIPN9YPUyEiBq+6qk7UY0Ew+vebWSCcmKelWkopuQVaonEL4258E5EljnFlvp+AStRNdAoA=='
#container_name = 'monitormate'

#create a client to interact with blob storage
connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'

from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import urllib.parse
from flask import Flask, jsonify, request, session, redirect, url_for


ACCOUNT_NAME = account_name
ACCOUNT_KEY = account_key
CONTAINER_NAME = 'cresendev'

def generate_sas_token(blob_name):
    # Generate a SAS token
    sas_token = generate_blob_sas(
        account_name=ACCOUNT_NAME,
        container_name=CONTAINER_NAME,
        blob_name=blob_name,
        account_key=ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )

    blob_url_with_sas = f'https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}?{sas_token}'
    return blob_url_with_sas

def generate_sas_for_url(url):
    # Parse the URL to get the blob name
    blob_name = url.split(f'https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/')[-1]

    # Decode the URL-encoded blob name
    blob_name = urllib.parse.unquote(blob_name)

    # Generate the SAS token
    sas_url = generate_sas_token(blob_name)
    return sas_url

import os
import json
import uuid

def format_references(result):
    references = []
    file_mappings = session.get('file_mappings', {})  # Get the current file mappings or initialize it
    
    for document in result['source_documents']:
        # Extract relevant information from metadata
        filepath = document.metadata.get('source', 'Unknown source')
        filename = os.path.basename(filepath)  # Extracts the filename from the filepath
        page_content = document.page_content
        url = document.metadata.get('url', 'No URL available')

        # Generate SAS URL for the document
        sas_url = generate_sas_for_url(url)

        # Generate a unique identifier
        file_id = str(uuid.uuid4())
        # Map the file_id to the SAS URL
        file_mappings[file_id] = sas_url

        # Format the reference as a dictionary using file_id instead of the SAS URL
        reference = {
            "filename": filename,
            "text": page_content,
            "file_id": file_id  # Use the unique identifier here
        }
        references.append(reference)

    # Update the session with the new file mappings
    session['file_mappings'] = file_mappings

    # Convert the list of references to JSON
    references_json = json.dumps(references, indent=4)
    return references_json


@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    question = data.get('data')
    
    # Get this session's chat history, or an empty list if this is a new session
    chat_history = session.get('chat_history', [])
    
    response = qa({"question": question, "chat_history": chat_history})
    answer = response['answer']
    chat_history.append((question, answer))
    
    # Store the updated chat history in this session
    session['chat_history'] = chat_history
    
    # Get the references if needed
    # source_documents = response.get('source_documents', [])
    references = format_references(response)
    # Store the references in the session
    session['references'] = references
    
    new_page_link = url_for('citations', _external=True)
    
    # Format the response to include the clickable link with 'citations' as the anchor text
    # and apply the color style to the 'citations' text
    formatted_response = f"{answer}<br/><br/>For more details: <a href='{new_page_link}' target='_blank' style='color: #1e80a3;'>Citations</a>"
    
    # Return the formatted response as HTML
    return jsonify({"response": True, "message": formatted_response})

from flask import Flask, render_template, session, json
@app.route('/citations')
def citations():
    # Retrieve references from session, make sure to convert them back from JSON
    references_json = session.get('references', '[]')
    references = json.loads(references_json)
    
    # Render the new_page.html template with the references data
    return render_template('citations.html', references=references)

@app.route('/view_source_file/<file_id>')
def view_source_file(file_id):
    # Retrieve the file_mappings from the session
    file_mappings = session.get('file_mappings', {})
    # Get the file URL using the file ID
    file_url = file_mappings.get(file_id)

    if not file_url:
        return "File not found", 404  # or handle the error as appropriate

    # Render the file content using the file URL
    return render_template('view_source_file.html', file_url=file_url)


@app.route('/clear', methods=['POST'])
def clear_chat():
    # Clear the chat history for this session
    session['chat_history'] = []
    return jsonify({"response": True})

if __name__ == '__main__':
    app.run(debug=True)







