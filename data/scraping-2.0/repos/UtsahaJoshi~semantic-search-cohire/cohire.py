import streamlit as st
import database as db
import uuid
import base64
import time
import cohere
import os
import torch
import numpy as np
import pandas as pd
from typing import List


c1, c2, c3 = st.columns(3)
with c2:
    st.title('COHIRE')
st.markdown("***")
if "page" not in st.session_state:
    st.session_state.page = 0
if "similar_results" not in st.session_state:
    st.session_state.similar_results = [] 

def goback(): st.session_state.page = 0
def uploadprofile(): st.session_state.page = 1
def searchprofile(): st.session_state.page = 2

torchfy = lambda x: torch.as_tensor(x, dtype=torch.float32)

def get_similarity(target: List[float], candidates: List[float], top_k: int):
    candidates = torchfy(candidates).transpose(0, 1)
    target = torchfy(target)
    cos_scores = torch.mm(target, candidates)

    scores, indices = torch.topk(cos_scores, k=top_k)
    similarity_hits = [{'id': idx, 'score': score} for idx, score in zip(indices[0].tolist(), scores[0].tolist())]

    return similarity_hits


def prepare_data_for_embedding(data):
    texts = [] 
    for item in data:
        name = item['name']
        experience = item['experience']
        hobbies = item['hobbies']
        gender = item['gender']
        texts.append(name + ' ' + gender + ' ' + experience + ' ' + hobbies)
    return texts


placeholder = st.empty()

if st.session_state.page == 0:
    col1, col2, col3 = placeholder.columns([0.2, 0.2,0.45])
    with col2:
        job = st.button('I need a job!',on_click=uploadprofile)
    with col3: 
        hire = st.button('I want to hire!',on_click=searchprofile)

    st.markdown("")
    st.markdown("Welcome to Cohire, where job seekers and employers can connect through our platform powered by cutting edge Artificial Intelligence. Our multilingual semantic search model helps employers find the right candidates for their open positions, while job seekers can upload their profiles, including their background, hobbies, and resume, to be discovered by top employers.")
    st.markdown("At Cohire, we understand the importance of finding the right fit for both job seekers and employers. That is why we have created a platform that uses AI to match the right candidates with the right positions. So if you are a job seeker looking for your next opportunity or an employer searching for top talent, look no further than Cohire.")
    st.markdown("Let us help you find the perfect match with our innovative platform powered by Cohere.")

elif st.session_state.page == 1:
    key = str(uuid.uuid4())
    name = st.text_input('Name', '')
    gender = st.selectbox(
    "Gender",
    ('Male', 'Female', 'Others'))
    picture = st.camera_input("Take your photo")
    experience = st.text_area("Experience & Skills")
    hobbies = st.text_area("Hobbies & Interests")
    resume = st.file_uploader("Upload your resume", type=['pdf'])
    col1, col2, col3 = st.columns([0.5, 0.2,0.45])
    with col2:
        if st.button("SAVE NOW"):
            if picture:
                my_picture = base64.b64encode(picture.read()).decode('utf-8')
            if resume:
                my_resume = base64.b64encode(resume.read()).decode('utf-8')
            db.insert_data(key, name, gender, my_picture, experience, hobbies, my_resume)
            with st.spinner('Saving...'):
                time.sleep(1)
    st.button('< Go Back',on_click=goback)

elif st.session_state.page == 2:
    search = st.text_area("Search your potential employee!")
    col1, col2, col3 = st.columns([0.5, 0.2,0.45])
    similar_results = st.session_state.similar_results
    with col2:
        if st.button("SEARCH"):
            model_name: str = 'multilingual-22-12'
            COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
            co = cohere.Client(COHERE_API_KEY)
            data = db.fetch_all_data()
            data_for_embed = prepare_data_for_embedding(data)
            response = co.embed(texts=data_for_embed, model='multilingual-22-12')
            embeddings = response.embeddings # All text embeddings 
            vectors_to_search = np.array(
                co.embed(model=model_name, texts=[search], truncate="RIGHT").embeddings,
                dtype=np.float32,
            )
            result = get_similarity(vectors_to_search, candidates=embeddings, top_k=1)
            similar_results = {}
            dataframe = pd.DataFrame(data)
            output_fields: List[str] = [
                "key", "name", "gender", "picture", "experience", "hobbies", "resume"
            ]
            for index, hit in enumerate(result):
                similar_example = dataframe.iloc[hit['id']]
                similar_results[index] = {field: similar_example[field] for field in output_fields}
    st.session_state.similar_results = similar_results
    if similar_results:
        picture=base64.b64decode((similar_results[0]['picture']))
        st.image(picture)
        st.markdown('**Name**: ' + similar_results[0]['name'])
        st.markdown("**Gender**: " + similar_results[0]['gender'])
        st.markdown("**Skills & Experiences**: " + similar_results[0]['experience'])
        st.markdown("**Hobbies & Interests**: " + similar_results[0]['hobbies'])
        resume=base64.b64decode((similar_results[0]['resume']))
        st.download_button(
            label="Download Resume",
            data=resume,
            file_name='resume.pdf',
            mime='application/pdf',
        )


    st.button('< Go Back',on_click=goback)
