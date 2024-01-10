#######################################
# Loosely based on the Build conversational Apps Tutorial: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
#######################################
import common, common.environment, common.process, common.query

# environment stuff
from dotenv import find_dotenv
from common.environment import load_env, create_new_es_index, load_session_vars, build_app_vars, env_list

# data chunking and loading stuff
from common.process import get_connections, extract_text_from_upload, load_document_text

# LLM interaction stuff
from common.query import ask_es_question, get_openai_callback, load_conv_chain 
from langchain.chains.question_answering import load_qa_chain
import openai

# UI stuff
from elasticsearch import Elasticsearch
import eland as ed
import json
import streamlit as st
from streamlit_chat import message


# load the environment file if needed.
if not 'env_file_loaded' in st.session_state:

    # upload variables from the .env file as environment variables
    env_path = find_dotenv()
    env_vars = dict(load_env(env_path=env_path))

    # load our values from the .env file into the session state.
    # only add our values, not the entire environment.
    load_session_vars(os_env_vars=env_vars, key_list=common.environment.env_list)
    build_app_vars()

    # don't load this multiple times
    st.session_state['env_file_loaded'] = True

st.title("ChatGPT-like clone")

def extract_es_index_fields(mapping_dict):
    fields = []
    
    # Check if the 'mappings' and 'properties' keys exist
    mappings = mapping_dict.get('mappings', {})
    properties = mappings.get('properties', {})
    
    for field_name, field_attributes in properties.items():
        fields.append(field_name)
        
        # Handle nested fields
        if 'properties' in field_attributes:
            nested_fields = extract_es_index_fields({'mappings': {'properties': field_attributes['properties']}})
            for nested_field in nested_fields:
                fields.append(f"{field_name}.{nested_field}")
                
    return fields


es = Elasticsearch(st.session_state.es_url)

# Get list of all indices
indices = es.indices.get_alias().keys()
non_system_indices = [index for index in indices if not index.startswith('.')]

# Convert to a list and sort it if needed
sorted_indices = sorted(list(non_system_indices))
option = st.selectbox("All indices:", sorted_indices)
mapping = es.indices.get_mapping(index=option)

st.write('You selected:', option)
st.write('Mapping:', mapping)
fields = extract_es_index_fields(mapping[option])
st.write('Fields:', fields)

# query = {
#     "_source": fields,  # Fetch only the fields we're interested in
#     "query": {
#         "match_all": {}  # Match all documents
#     }
# }

# # Execute the query
# response = es.search(index=option, body=query)
# st.write(response)

df = ed.DataFrame(st.session_state.es_url, option)
dft = df[['text']]
st.dataframe(dft, hide_index=True)

st.divider()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})