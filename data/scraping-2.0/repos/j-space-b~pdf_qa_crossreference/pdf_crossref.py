import os, tempfile
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chains import SequentialChain
from langchain.retrievers import WikipediaRetriever 
from langchain.chains import SimpleSequentialChain

# define which LLM model to use - this is constructed for OpenAI
import datetime
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo" 
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# Streamlit app
st.subheader('PDF Summarize and Cross Reference Tool')

# Get OpenAI API key and source pdf input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
    st.caption("*If you don't have an OpenAI API key, get one [here](https://platform.openai.com/account/api-keys).*")
source_doc = st.file_uploader("Source Document", label_visibility="collapsed", type="pdf")

# If the 'Ask' button is clicked
if st.button("Summarize and Cross Reference"):
    # Validate inputs
    if not openai_api_key.strip() or not source_doc:
        st.error(f"Please provide the missing fields/keys.")
    else:
        try:
            with st.spinner('Thinking...'):
              # Save uploaded file temporarily to disk, load and split the file into pages
              with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                  tmp_file.write(source_doc.read())
              loader = PyPDFLoader(tmp_file.name)
              pages = loader.load_and_split()
              os.remove(tmp_file.name)

              # Create embeddings for the pages and insert into Chroma database
              embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
              vectordb = Chroma.from_documents(pages, embeddings)

              # Initialize the OpenAI module, load and run the summarize chain
              llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

              # prompt 1
              prompt1 = ChatPromptTemplate.from_template(
                  "Write a summary within 300 words, emphasize intended \
                  competitive advantage."
              )

              # Chain 1
              chain1 = load_summarize_chain(llm=llm, chain_type="stuff", output_key="summary") 
              summary = chain1.run(input_documents=pages, question=prompt1) 
              
              
              # prompt 2
              prompt2 = ChatPromptTemplate.from_template(
              "Lookup the top 5 competitors against a product that is \
              summarized by {summary} for the same demographic market \
              and list the top 3 competitive advantages for the first product in \
              relation to the competitors.  Also list the top 3 competitive \
              disadvantages for the first product and what could be developed\
              to be more competitive"
              )
              
              # chain 2
              formatted_prompt2 = prompt2.format(summary=summary) 
              chain2 = LLMChain(llm=llm, prompt=formatted_prompt2, output_key="analysis") 


              # master chain
              full_output = SequentialChain(
                chains=[chain1, chain2],
                input_documents=pages, 
                question=prompt1,
                input_variables=["prompt1"],
                output_variables=["summary", "analysis"],
                verbose=True
              )

              st.success(full_output) 
        except Exception as e:
            st.exception(f"An error occurred: {e}")
