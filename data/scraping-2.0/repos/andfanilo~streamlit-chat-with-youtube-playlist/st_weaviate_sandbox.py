"""
For exploring Weaviate index
"""
import openai
import streamlit as st
import weaviate

st.set_page_config(initial_sidebar_state="collapsed")

openai.api_key = st.secrets["OPENAI_API_KEY"]

weaviate_index = "LlamaIndex"

st.title("#️ Explore Weaviate LlamaIndex")


@st.cache_resource(show_spinner=False)
def load_weaviate_client():
    client = weaviate.Client(
        url=st.secrets["WEAVIATE_URL"],
        auth_client_secret=weaviate.AuthApiKey(api_key=st.secrets["WEAVIATE_API_KEY"]),
        additional_headers={"X-OpenAI-Api-Key": st.secrets["OPENAI_API_KEY"]},
    )
    return client


client = load_weaviate_client()

#######################################
### SIDEBAR
#######################################

with st.sidebar:
    index_name = st.text_input("Weaviate Class", weaviate_index)

    if not client.schema.exists(index_name):
        st.error("Weaviate class does not exist")
        st.stop()

    unlock_button = not st.toggle("Unlock Delete button")
    if st.button(f"Delete {index_name} index", disabled=unlock_button):
        client.schema.delete_class(index_name)
        st.toast(f"{index_name} deleted", icon="🗑️")


#######################################
### APP
#######################################


st.subheader("Generic info")

with st.expander(f"{index_name} schema"):
    st.json(client.schema.get(), expanded=False)

with st.expander("Get random objects"):
    st.json(client.data_object.get(), expanded=False)

st.divider()

st.subheader("Search by ID")
include_vector = st.toggle("Include vectors", False)

object_id = st.text_input("Enter Data Object ID")
try:
    data_object = client.data_object.get_by_id(
        object_id,
        class_name=weaviate_index,
        with_vector=include_vector,
    )
    st.json(data_object, expanded=False)
except Exception as e:
    st.error(str(e), icon="☠️")

kw_query = st.text_input("Enter Keyword query", "CSS")
try:
    response = (
        client.query
        .get(weaviate_index, ["doc_id", "text"])
        .with_bm25(
            query=kw_query
        )
        .with_limit(3)
        .do()
    )
    st.json(response, expanded=False)
except Exception as e:
    st.error(str(e), icon="☠️")

with st.expander("Problematic Vector Search"):
    response = (
        client.query
        .get(weaviate_index, ["doc_id", "text"])
        .with_near_text({
            "concepts": ["CSS"]
        })
        .with_limit(3)
        .with_additional(["distance"])
        .do()
    )
    st.json(response, expanded=False)