import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index import Document
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
import openai
import os
from dotenv import load_dotenv
load_dotenv()
from llama_index import download_loader
st.set_page_config(layout='wide')

# Retrieve OpenAI API key from secrets

# Set up Streamlit app
st.title('DEMO, WIKI')
input_column, response_column = st.columns([2,3])

# Add inputs for Wikipedia pages
pages = input_column.text_input('Enter Wikipedia pages (comma-separated)')

formality = input_column.slider('Formality', 0, 100, 50)

# Add inputs for user context
user = input_column.text_input('Who are you?')
audience = input_column.text_input('Who is the audience?')

# Add inputs for user goal
goal = input_column.text_input('Prompt?')

# Add dropdown for response structure
structure = input_column.selectbox('Structure of the response', ['Summary', 'One-pager', 'Report', 'Speech', 'Presentation','Testament'])

# Add dropdown for summary length

language = input_column.selectbox('Language', ['English', 'Dutch'])

# Add slider for temperature
temperature = input_column.slider('Temperature', 0.0, 1.0, 0.5)

# Add slider for max tokens
max_tokens = input_column.slider('Max Tokens', 100, 2000, 500)

if pages:
    # Create the ServiceContext with the user-selected temperature
    service_context = ServiceContext.from_defaults(llm=OpenAI(temperature=temperature, model="gpt-4", max_tokens=max_tokens))
    pages = [page if not page.isdigit() or len(page) != 4 else f"{page} (year)" for page in pages.split(',')]

    with st.spinner('Loading Wikipedia articles...'):
        WikipediaReader = download_loader("WikipediaReader")
        loader = WikipediaReader()
        documents = loader.load_data(pages=pages)

        index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    if input_column.button('Generate'):
        # Determine formality phrase
        if formality < 33:
            formality_phrase = "In a casual and conversational style, "
        elif formality < 67:
            formality_phrase = "In a neutral style, "
        else:
            formality_phrase = "In a highly formal and academic style, "

        # Add user context and structure to the query
        query = f"As {user}, I need a {structure.lower()} of the document for {audience}. My prompt is {goal}. Please provide the response in markdown format with appropriate features. {formality_phrase}Generate the {structure.lower()}."

        # Generate the response
        with st.spinner(f'Generating {structure.lower()}...'):
            retriever = VectorIndexRetriever(index=index)
            query_engine = RetrieverQueryEngine(retriever=retriever)
            response = query_engine.query(query)
            # Store the response in the session state
            st.session_state['response'] = response

    # Display the response stored in the session state
    if 'response' in st.session_state:
        response_column.markdown(st.session_state['response'])

input_column.markdown("<p style='text-align: center;'> Wiki DEMO </p>", unsafe_allow_html=True)