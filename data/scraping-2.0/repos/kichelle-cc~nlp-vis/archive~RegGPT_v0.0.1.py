import os
import openai
from PIL import Image
import streamlit as st
openai.api_key = st.secrets["OPENAI_API_KEY"]
from llama_index import LLMPredictor, PromptHelper, GPTSimpleVectorIndex
from llama_index.logger import LlamaLogger
from langchain import OpenAI
from llama_index.composability import ComposableGraph
# this code block creates indivisual indexes for each of the documents. We will be using vector store index (Ref:)
# but first define the model parameters that will be used for that chatbot
from langchain.chat_models import ChatOpenAI
# chunk size - size of each context chunk
chunk_size_limit = 3500

# max input size - max number of input prompt token (includes the query + context chunk + output)
max_in = 4096

# max output token - max number of tokens the chatbot will return as response
num_out = 600

# temp - crrativity vs reproducability. Setting as 0 to reduce hallucination
temp = 0

# note max_in + max_out = context length of the model. FOr davincii it is 4097
# note chunk_size + our wuery  =  max_in
# model name - using text-davinci-003 as it is optimised for instructions
model_name = "text-davinci-003"


# debugg #
import tempfile
cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache") 
print(cache_dir)
# debugg #


llm_predictor = LLMPredictor(llm=OpenAI(temperature=temp, model_name=model_name))
prompt_helper = PromptHelper(max_input_size = max_in, num_output=num_out, max_chunk_overlap=200)
from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper,chunk_size_limit=chunk_size_limit)
graph = ComposableGraph.load_from_disk('reg_v8_fast.json', service_context = service_context)


# function for query and response
def ask_ai(prompt):
    query_configs = [
      {
          "index_struct_type": "simple_dict",
          "query_mode": "default",
          "query_kwargs": {
              "similarity_top_k": 2,
          },
      },
      {
          "index_struct_type": "simple_dict",
          "query_mode": "default",
          "query_kwargs": {
              "similarity_top_k": 1,
          }
      },
             ]
    
    index = graph
    response = index.query(prompt+"Only refer to the context information. If the context information is not helpful say Unfortunately the context information does not contain the answer to your question.", query_configs)
    return response


# streamlit frontend code


image = Image.open('deloitte-logo-white.png')
st.sidebar.image(image)
st.sidebar.title("RegGPT V0.0.1")


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
def get_text():
    input_text = st.text_input("Ask me a question: ", key = "input")
    return input_text

user_input = get_text()

if user_input:
    output = ask_ai(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.session_state['past'][i]
        st.session_state['generated'][i]
