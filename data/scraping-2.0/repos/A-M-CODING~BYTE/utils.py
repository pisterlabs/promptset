import streamlit as st
import cohere 
import requests
from cohere.responses.classify import Example
from google.cloud import vision
import os
import json
import hashlib
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredAPIFileIOLoader
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env
import toml

# Access the credentials
config = st.secrets["google_credentials"]

# Construct a credentials object from the dictionary
credentials = service_account.Credentials.from_service_account_info(config)


cohereAPIKey = os.getenv("COHERE_API_KEY")
googleSearchID = os.getenv("GOOGLE_SEARCH_ID")
searchEngineID = os.getenv("SEARCH_ENGINE_ID")

# WEAVIATE

# API keys and endpoints
WEAVIATE_ENDPOINT = os.getenv("WEAVIATE_ENDPOINT")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Weaviate and Cohere clients
client = weaviate.Client(
    url=WEAVIATE_ENDPOINT,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    additional_headers={"X-Cohere-Api-Key": COHERE_API_KEY}
)
co = cohere.Client(COHERE_API_KEY)


# VERIFY TENANT
# Function to verify if a tenant exists in Weaviate
def verify_tenant(tenant_name):
    tenants = client.schema.get_class_tenants(class_name='UserInformation')
    return tenant_name in [t.name for t in tenants]

def show_user_documents_screen(tenant_id):
    st.subheader("Your Uploaded Documents")

    if not tenant_id:
        st.error("Please log in to view your documents.")
        return

    documents = get_objects_for_tenant(tenant_id)
    if documents:
        for index, doc in enumerate(documents):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.text(f"Source: {doc.get('source')}")
                # Use the index to create a unique key for each text area
                st.text_area("", value=doc.get('content'), height=100, disabled=True, key=f"doc_{index}")

# Function to create a new tenant in Weaviate
def create_new_tenant(tenant_name):
    client.schema.add_class_tenants(
        class_name='UserInformation', 
        tenants=[weaviate.Tenant(name=tenant_name)]
    )


def get_objects_for_tenant(tenant_name):
    try:
        results = (
            client.query.get('UserInformation', ['content', 'source'])
            .with_tenant(tenant_name)
            .do()
        )
        return results.get('data', {}).get('Get', {}).get('UserInformation', [])
    except Exception as e:
        print(f"Error retrieving objects for tenant {tenant_name}: {e}")
        return []

# HASHED PASSWORD
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# UPLOAD DOCS

def upload_document(document, tenant_id):
    unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
    loader = UnstructuredAPIFileIOLoader(
        file=document,
        metadata_filename=document.name,
        api_key=unstructured_api_key,
        mode="elements",
        strategy="auto",
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter)

    data_objects = []
    for doc in docs:
        data_object = {
            "content": doc.page_content,
            "source": doc.metadata.get("filename"),
            "tenant": tenant_id
        }
        data_objects.append(data_object)
    return data_objects

def batch_import_documents(data_objects, tenant_name):
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for obj in data_objects:
            batch.add_data_object(
                data_object=obj,
                class_name="UserInformation",
                tenant=tenant_name
            )








#DETECT TEXT
def detect_text(uploaded_file):
    """Detects text in the uploaded file."""
    client = vision.ImageAnnotatorClient(credentials=credentials)

    # Read content directly from the uploaded file-like object
    content = uploaded_file.getvalue()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        # Extract the entire block of text
        full_text = texts[0].description
        print('Extracted Text:\n')
        print(full_text)
        return full_text
    else:
        print('No text found')
        return None
    

# ALTERNATIVE PRODUCTS
def get_links(nutr_label, user_info):
  response = co.chat( 
    model='command',
    message=f'links to healthier alternate food products for food item with this nutritional label: {nutr_label}, and tailored for this user: {user_info}',
    search_queries_only=True
  ) 

  search_query_text = response.search_queries[0]['text']

  print(search_query_text)


  searchQuery = f"links to buy food products for {search_query_text}"
  url = "https://www.googleapis.com/customsearch/v1"

  params  = {
      "q": searchQuery,
      "key": googleSearchID,
      "cx": searchEngineID
  }

  response = requests.get(url, params=params)
  results = response.json()

  links_array = [item['link'] for item in results['items']]

  for i in enumerate(links_array):
      print(i)

  return links_array


def get_responses(responses, tenant_id):
    data_object = {
            "user_form": responses,
            "source": "User's Form",
            "tenant": tenant_id
        }
    return data_object

def import_responses(data_object, tenant_name):
    try:
        response = client.data_object.create(
            class_name='UserInformation',  
            data_object=data_object,
            tenant=tenant_name
        )
        # Save the generated ID in the session state
        st.session_state["form_object_id"] = response
        return response
    except Exception as e:
        print(f"Error storing user form data: {e}")
        return None

    
def get_info_for_tenant(tenant_name, object_id):
    try:
        data_object = client.data_object.get_by_id(
            uuid = f'{object_id}',
            class_name='UserInformation',
            tenant=tenant_name
        )
        return data_object
    except Exception as e:
        print(f"Error retrieving user form data: {e}")
        return None
 # Function to send tenant_name to Flask app
def set_tenant_in_flask(tenant_name):
    url = "https://byteapp-ltle5vf4cq-el.a.run.app/set_tenant"  # Replace with your actual Flask app URL
    data = {"tenant_name": tenant_name}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            st.success("Tenant name set successfully in Flask.")
        else:
            st.error(f"Failed to set tenant name in Flask. Status code: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    
