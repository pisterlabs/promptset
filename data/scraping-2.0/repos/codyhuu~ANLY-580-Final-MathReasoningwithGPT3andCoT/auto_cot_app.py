import streamlit as st
import time
import json
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from auto_cot_utils import *
import re


"""
# Math Reasoning & Inference - By GPT-3 and Auto-Chain of Thoughts

🔧 Made by Zheyuan Hu, Muwen You, Ruobing Yan, Dian Zhi

📝 Solve math problems
"""

if 'options' not in st.session_state:
    st.session_state['options'] = ''
if 'results' not in st.session_state:
    st.session_state['results'] = ''

@st.cache(allow_output_mutation=True)
def init_variables():

    # load train question embeddings
    train_q_embeddings = np.load('train_q_embeddings.npy')

    # select 3 as the number of k-means clusters
    K = 3 

    # load train question embedding centroids
    cluster_centers = np.load('train_q_cluster_centers.npy')

    # load train data
    train_data = read_jsonl('train.jsonl')

    # find the most representative training questions indices
    most_repr_indices = {}
    for c in range(K):
        most_repr_indices[c] = get_nearest_embeddings_idx(cluster_centers[c, :], train_q_embeddings)

    # initialize the pretrained sentence bert model
    sbert = SentenceTransformer('all-mpnet-base-v2')

    return sbert, most_repr_indices, train_data, train_q_embeddings, cluster_centers

def generate_repr_cot(gpt3_engine, train_data, most_repr_indices):
    repr_auto_cot = {}
    for key, value in tqdm(most_repr_indices.items()):
        q = get_qa_by_idx(value, train_data)[0]
        prompt = format_prompt(q, keywords=True)
        msg, completion = get_gpt3_response_text(prompt, engine=gpt3_engine)
        repr_auto_cot[key] = prompt + completion + '\n\n'
    return repr_auto_cot


def init_gpt3_qa_system(gpt3_engine, repr_auto_cot, sbert, most_repr_indices, train_data, train_q_embeddings, cluster_centers):
    model = GPT3ArithmeticReasoning(gpt3_engine, sbert, repr_auto_cot, most_repr_indices, train_data, train_q_embeddings, cluster_centers)
    return model

@st.cache(allow_output_mutation=True)
def get_test_examples():
    test_data = read_jsonl('test.jsonl')
    return test_data

col1, col2 = st.columns(2)
with col1:
    engine = st.radio('Engine',('text-davinci-003', 'text-curie-001'))
with col2:
    zs = st.checkbox('0-shot')
    zsk = st.checkbox('0-shot with keywords')
    acr = st.checkbox('Auto-COT most repr question')
    acn = st.checkbox('Auto-COT nearest question')
    mcr = st.checkbox('Manual-COT most repr question')
    mcn = st.checkbox('Manual-COT nearest question')

api_key = st.text_input('Enter your OpenAI token here', type='password')

input_question = st.text_area('Your math reasoning question here', key= 'options')

def run_callback():
    with st.spinner('Working...'):
        set_openai_apikey(api_key)
        sbert, most_repr_indices, train_data, train_q_embeddings, cluster_centers = init_variables()
        repr_auto_cot = generate_repr_cot(engine, train_data, most_repr_indices)
        model = init_gpt3_qa_system(engine, repr_auto_cot, sbert, most_repr_indices, train_data, train_q_embeddings, cluster_centers)
        prompt_methods = {
            '0-shot': zs, 
            '0-shot with keywords': zsk, 
            'Auto-COT representative question': acr, 
            'Auto-COT nearest question': acn, 
            'Manual-COT representative question': mcr, 
            'Manual-COT nearest question': mcn
        }
        results = ''
        for key, val in prompt_methods.items():
            if val:
                prompt = model.generate_gpt3_prompt(input_question, prompt_method=key)
                completion = model.get_gpt3_completion()
                results += '\n\n' + '=' * 80 + '\n\nPrompting Method: ' + key + '\n\nPrompt: ' \
                    + prompt + '\n\nCompletion: ' + completion[1] + '\n\n'
                results = re.sub(r'$', '', results)
                results = re.sub(r'#', '', results)
        st.session_state['results'] = results

def button1_callback():
    test_data = get_test_examples()
    st.session_state['options'] = test_data[3]['question']
def button2_callback():
    test_data = get_test_examples()
    st.session_state['options'] = test_data[1]['question']

c1, c2, c3 = st.columns(3)
with c1:
    st.button('Run', on_click = run_callback)
with c2:
    st.button('Example Question 1', on_click = button1_callback)
with c3:
    st.button('Example Question 2', on_click = button2_callback)

container = st.container()
container.markdown('**Answer:**')
container.markdown(st.session_state['results'])
