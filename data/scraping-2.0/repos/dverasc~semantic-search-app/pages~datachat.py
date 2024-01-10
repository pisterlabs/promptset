
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import streamlit as st
from langchain.llms import Ollama


from langchain.llms import Ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# from functions import conversational_chat
from streamlit_chat import message
from langchain.vectorstores import Qdrant
from qdrant_client.http import models as qdrant_models




st.set_page_config(
    page_title="Data Chat"
)


st.text("ollama initialize...")
ollama = Ollama(base_url='http://localhost:11434',
model="llama2")


st.markdown(
"""
**The final feature I'm including in this demo is a Q&A bot aka the hello world of LLMs. This particular bot has custom prompting to talk to users in Spanish within the context of e-commerce product data** 
"""
)
# search qdrant
collection_name = "amazon-products"

client = QdrantClient('http://localhost:6333')
# Initialize encoder model
model = SentenceTransformer('all-MiniLM-L6-v2')




if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hola ! Estoy aqui para responder sobre cualquier preguntas que tengas sobre:  " + collection_name + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hola ! 👋"]
    

# docs = hits
# prompt = st.text_input("Entrar tu pregunta aqui")


def conversational_chat(query):
    vector = model.encode(query).tolist()
    hits = client.search(
    collection_name="amazon-products",
    query_vector=vector,
    limit=3
)
    toplevelkeys = ['Product 1','Product 2','Product 3','Product 4','Product 5']
    context = {'Product 1': [], 'Product 2': [], 'Product 3': [],  'Product 4': [], 'Product 5': []}
    count = 0
    for hit in hits:
    #     print(hits)
    #     d[count] = i
        count+=1
        key1 = "Price"
        key2 = "About Product"
        key3 = "Product Name"
        key4 = "Product Specification"
        val1 = hit.payload["Selling Price"]
        val2 = hit.payload["About Product"]
        val3 = hit.payload["Product Name"]
        val4 = hit.payload["Product Specification"]
        for i in toplevelkeys:
            hitinstance = {key1:val1,key2:val2,key3:val3,key4:val4}
            context[i].append(hitinstance)
    input_prompt = f"""[INST] <<SYS>>
    You are a customer service agent for a latin american e-commerce store. As such you must always respond in the Spanish language. Using the search results for context: {context}, do your best to answer any customer questions. If you do not have enough data to reply, make sure to tell the user that they should contact a salesperson. Everytime you don't reply in Spanish, you will be punished
    <</SYS>>

    {query} [/INST]"""
    output = ollama(input_prompt)
    return output

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Puedes hablar sobre los productos de la tienda de e-commerce aqui (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

