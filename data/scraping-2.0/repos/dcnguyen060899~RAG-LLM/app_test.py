import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
# Import torch for datatype attributes
import torch
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext
# Import deps to load documents
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from pathlib import Path
import pypdf
import time
import os
import tempfile

# Define variable to hold llama2 weights namingfiner
name = "gpt2"
# Set auth token variable from hugging face
auth_token = "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"

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(name, use_auth_token=auth_token)

    # Create model
    model = GPT2LMHeadModel.from_pretrained(name)
    
    return model, tokenizer

model, tokenizer = get_tokenizer_model()

# disclaimer
st.title('LLM Deployment Prototype for Production')
st.caption("Special thanks to my mentor, Medkham Chanthavong, for all the support in making this project a reality. Connect with Medkham on LinkedIn: [Medkham Chanthavong](https://www.linkedin.com/in/medkham-chanthavong-4524b0125/).")

st.subheader('Disclaimer')
st.write("Due to Streamlit's lack of cloud-GPU support and limited memory, we're using GPT-2 instead of Llama 2 7B. GPT-2, \
being smaller, is less capable of accurately retrieving information from external data. This site is a prototype for LLM deployment \
and may not retrieve precise data from external sources. Additionally, the site might experience crashes due to memory constraints.\
In case of a crash, please feel free shoot me a message at my [LinkedIn profile](https://www.linkedin.com/in/duwe-ng/) so I can reboot the site. \
For a more advanced model demonstration using Llama 2 7B, check out our Colab notebook: \
[LLM Deployment Prototype](https://colab.research.google.com/drive/1bGf9rKntMjH4KtpKs9ryucj1nbiKs_zk?usp=sharing).")


# # Initialize the SimpleInputPrompt with an empty template
# query_wrapper_prompt = SimpleInputPrompt("{query_str}")

# # Streamlit UI to let the user update the system prompt
# # Start with an empty string or a default prompt
# default_prompt = ""
# user_system_prompt = st.text_area("How can I best assist you?", value="", height=100)
# update_button = st.button('Request')


# Import the prompt wrapper for llama index
# Create a system prompt 
system_prompt = """
Hey there! 👋 You are here to help you out in the best way you can. Think of yourself as your friendly and trustworthy assistant. Your top priority is to be super helpful and safe in your responses. \
Here's a quick heads up about what you can expect from me: \
1. Positive Vibes Only: I steer clear of anything harmful, unethical, or offensive. No racism, sexism, or toxic stuff here. I'm all about being respectful and keeping things on the up-and-up. \
2. Making Sense Matters: If the user question seems a bit confusing or doesn't quite add up, You will let you know and try to clarify things instead of giving the user a misleading answer. \
3. Honesty is Key: Not sure about something? You won't make stuff up. If you don't have the answer, you will be upfront about it. \
4. All About Your PDFs: The user documents are your focus. Got a question about the content in your PDFs? That's exactly what you are here for. Let's dive into those documents and find the answers the user need! \
So, how can you assist the user today with their PDFs?
"""

# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str}")

# Initialize the llm object with a placeholder or default system prompt
llm = HuggingFaceLLM(
    context_window=1024,
    max_new_tokens=128,
    system_prompt="",
    query_wrapper_prompt=query_wrapper_prompt,
    model=model,
    tokenizer=tokenizer
)

# # Function to update the system prompt and reinitialize the LLM with the new prompt
# def update_system_prompt(new_prompt):
#     global llm
#     llm.system_prompt = new_prompt


# if update_button:
#     # Update the system prompt and reinitialize the LLM
#     update_system_prompt(user_system_prompt)
#     st.success('Requested')
    
# check if user request save to the memory
# st.write(llm.system_prompt)

# Create and dl embeddings instance
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embeddings
)

# And set the service context
set_global_service_context(service_context)


# Upload PDF and process it
st.subheader("Please upload your data to enable the retrieval system; otherwise, it will not respond.")

documents = []
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write the uploaded file to a file in the temporary directory
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Now you can use SimpleDirectoryReader on the temp_dir
        documents = SimpleDirectoryReader(temp_dir).load_data()
        st.success(f"File {uploaded_file.name} uploaded successfully.")
        
index = VectorStoreIndex.from_documents(documents)
        
# Setup index query engine using LLM
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

# Create centered main title
st.title('👔 HireMind 🧩')
        
# setup a session to hold all the old prompt
if 'messages' not in st.session_state:
    st.session_state.messages = []
        
# print out the history message
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
        
        
# Create a text input box for the user
# If the user hits enter
prompt = st.chat_input('Input your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    response = query_engine.query(prompt)
    
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append(
        {'role': 'assistant', 'content': response}
    )

