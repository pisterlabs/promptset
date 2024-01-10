import os, streamlit as st
from PIL import Image
# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= ""

from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI
from llama_index import StorageContext, load_index_from_storage
image = Image.open('logo.png')
# Define a simple Streamlit app
st.image(image, width=100)

query = st.text_input("What would you like to ask? (source: data/)")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
   
            # these lines load a raw doc directly this is handled by another prog now
            # Load documents from the 'data' directory
            #documents = SimpleDirectoryReader('data').load_data()
           
           # Rebuild storage context

            storage_context = StorageContext.from_defaults(persist_dir="")

            # Load index from the storage context
            new_index = load_index_from_storage(storage_context)

            new_query_engine = new_index.as_query_engine()
            response = new_query_engine.query(query)
            #print(response)
            
            query_engine = new_index.as_query_engine()
            response = query_engine.query(query)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
