import os
import glob
import json
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from streamlit_ace import st_ace
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from PIL import Image

os.environ['OPENAI_API_KEY'] = "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"

llm = OpenAI(temperature=0, verbose=True)
embeddings = OpenAIEmbeddings()

# Specify the folder path containing PDF files
folder_path = "C:/Users/caten/OneDrive/Desktop/ACN"

# Find all PDF files in the folder
pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

loaders = []
for pdf_file in pdf_files:
    loaders.append(PyPDFLoader(pdf_file))

all_pages = []

# Load and split the PDFs
for loader in loaders:
    pages = loader.load_and_split()
    all_pages.extend(pages)
store = Chroma.from_documents(all_pages, embeddings, collection_name='immigrant_re')

vectorstore_info = VectorStoreInfo(
    name="immigrant_report",
    description="african immigrants report as a pdf",
    vectorstore=store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

with open('data.json', 'r') as json_file:
    document_info = json.load(json_file)

image_path = "ACN_LOGO.webp"  # Replace this with your image file path

# Set the desired width for the image (in pixels)
desired_width = 100

image = Image.open(image_path)

# Resize the image to the desired width while maintaining the aspect ratio
aspect_ratio = image.width / image.height
desired_height = int(desired_width / aspect_ratio)
resized_image = image.resize((100, 50))

# Create a sidebar for the title
st.sidebar.image(image_path, caption='ACN', use_column_width=True)
st.sidebar.title('African Collaborative Network(ACN)')

# Chat-like interface for user interaction
st.title('ACN GPT Analyst')

# List of suggestion queries
suggestions = ["Tell me about African immigration history in the US", "Challenges faced by African immigrants in the United States", "Immigration policies", "Statistics on African immigrants in the US"]

# Initialize a boolean variable to track if a query is being typed
query_typing = False

# Initialize a dictionary to store search results
search_results_cache = {}

# Initialize chat history for typed queries using SessionState
if 'typed_query_history' not in st.session_state:
    st.session_state.typed_query_history = []

# Function to display query response with additional information
def display_query_response(query, response):
    response_color = "green"  # You can change this to the desired color
    
    st.markdown(f'<span style="color: {response_color};"><strong>Response for \'{query}\':</strong></span>', unsafe_allow_html=True)
    
    # Check if the response contains meaningful information or is a generic "I don't know"
    if "I don't know" in response:
        st.write("I don't have that information right now.")
    else:
        st.write(response)
        search = store.similarity_search_with_score(query)
        a = [(search[0][0].metadata)]
        for m in a:
            source_path = m.get("source", None)
            selected_document = document_info.get(source_path, {})
            st.markdown('<span style="color: green;">**Research paper info:**</span>', unsafe_allow_html=True)
            st.markdown(f"**Title:** {selected_document.get('title', 'N/A')}")
            st.write(f"**Citation:** {selected_document.get('citation', 'N/A')}")
            st.write(f"**web Link:** {selected_document.get('pdf_link', 'N/A')}")
    st.write("---")

# Display suggestion buttons
for suggestion in suggestions:
    if st.button(suggestion):
        # Set the query_typing variable to False when a suggestion is clicked
        query_typing = False

        # Check if the response is already in the cache
        if suggestion in search_results_cache:
            response = search_results_cache[suggestion]
        else:
            # If not in the cache, run the query and store the response
            response = agent_executor.run(suggestion)
            search_results_cache[suggestion] = response

        # Display the full answer, including additional information
        display_query_response(suggestion, response)

# Text input for user queries
user_query = st.text_input('Ask me about african immigrants and refugees in USA:', '')

# Check if the user is typing a query
if user_query:
    query_typing = True

# Process user's typed query if not typing a suggestion
if query_typing:
    # Check if the response is already in the cache
    if user_query in search_results_cache:
        response = search_results_cache[user_query]
    else:
        # Split the user query into smaller chunks to fit within the model's limit
        chunked_queries = [user_query[i:i+4096] for i in range(0, len(user_query), 4096)]

        # Initialize an empty response
        response = ""

        for chunked_query in chunked_queries:
            # If not in the cache, run the query and append the response
            chunked_response = agent_executor.run(chunked_query)
            search_results_cache[chunked_query] = chunked_response
            response += chunked_response

        # Add user query to typed query history if it's not a duplicate
        if not any(entry["query"] == user_query for entry in st.session_state.typed_query_history):
            st.session_state.typed_query_history.append({"query": user_query, "response": response})

    # Display the full answer, including additional information
    display_query_response(user_query, response)

# Display chat history in the sidebar
st.sidebar.title('Query History')
clear_all_typed_query_history = st.sidebar.button("Clear Query History")

if clear_all_typed_query_history:
    st.session_state.typed_query_history = []  # Clear all typed query history

for i, entry in enumerate(st.session_state.typed_query_history):
    query = entry["query"]
    response = entry["response"]
    if st.sidebar.button(f"{i + 1}. {query}", key=f"typed_query_history_button_{i}"):
        st.write(f"Response for '{query}':")
        st.write(response)

