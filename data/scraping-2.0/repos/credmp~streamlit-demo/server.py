#!/usr/bin/env python3

import os, streamlit as st

# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= ""

from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from langchain.llms.openai import OpenAI

# Define a simple Streamlit app
st.title("Welcome to hackernotes")
query = st.text_input("What would you like to ask?", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            # This example uses text-davinci-003 by default; feel free to change if desired
            llm_predictor = LLMPredictor(
                llm=OpenAI(temperature=0, model_name="text-davinci-003")
            )

            # Configure prompt parameters and initialise helper
            max_input_size = 4096
            num_output = 1000
            max_chunk_overlap = 0.20

            prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

            # Load documents from the 'data' directory
            print("Loading documents...")
            documents = SimpleDirectoryReader("data", recursive=True).load_data()
            print(f"Loaded {len(documents)} documents.")
            service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor, prompt_helper=prompt_helper
            )
            index = GPTVectorStoreIndex.from_documents(
                documents, service_context=service_context
            )

            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
