import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
import pinecone
from openai_helper import create_prompt, generate_answer
from insert_pinecone_docs import \
    insert_pinecone_file_path ,insert_pinecone_S3
from pinecone_text.sparse import BM25Encoder
import pandas as pd

from config import PINECONE_API_KEY, \
PINECONE_REGION, PINECONE_INDEX_NAME, \
MODEL_NAME, MODEL_NAME_CLIP,IMAGES_PATH

@st.cache_resource
def get_model_for_documents(MODEL_NAME:str) -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)
    return model

@st.cache_resource
def get_model_for_images():
    model = SentenceTransformer(MODEL_NAME_CLIP)
    return model

st.set_page_config(page_title="Search Engine", page_icon="🔍", layout="wide")

search_upload = st.sidebar.radio("Select the Analysis Type", \
                                 ('Search', 'Upload Docs'))

if search_upload == 'Upload Docs':

    upload_category = st.selectbox('Select the Upload Type', ['S3', 'Local File Path'])
    if upload_category == 'S3':
        s3_bucket_name = st.text_input('Enter the S3 bucket here:')
        CATEGORY = st.text_input('Enter the category here:')
        if st.button('Submit'):
            insert_pinecone_S3(s3_bucket_name,CATEGORY)
    else:
        file_path = st.text_input('Enter the file path here:')
        CATEGORY = st.text_input('Enter the category here:')
        if st.button('Submit'):
            insert_pinecone_file_path(file_path,CATEGORY)
else:

    st.sidebar.title('Analysis Parameters')

    selected_category = st.sidebar.selectbox('Select the Analysis Type',
                    ['Documents', 'Image'])

    if selected_category == 'Image':
        selected_analysis_category = st.sidebar.selectbox('Select the Analysis Category',["Text","Image"])

    if selected_category == 'Documents':
        st.header('Search Engine - Document')

        st.markdown(" Input the Text and click Submit"       )
        user_input = st.text_input('Enter your question here:')
        CATEGORY = st.text_input('Enter the category here:')


        if st.button('Submit'):
            model = get_model_for_documents(MODEL_NAME)
            xq = model.encode([user_input]).tolist()
            length = 512 - 384 
            list_custom = [1] * length
            embedding_all = xq + list_custom
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_REGION)
            index = pinecone.Index(PINECONE_INDEX_NAME)
            
            if CATEGORY == '':
                xc = index.query(embedding_all, top_k=5,
                                include_metadata=True)
            else:
                xc = index.query(embedding_all, top_k=5,
                             filter={"category": {"$eq": CATEGORY}},
                                include_metadata=True)
            
            contexts = [
                x['metadata']['sentence'] for x in xc['matches']
            ]
            context= "\n\n".join(contexts)
            prompt = create_prompt(context,user_input)    
            reply = generate_answer(prompt)
            st.write(reply)

    elif selected_category == 'Image':
        st.header('Image Search Engine')

        if selected_analysis_category == 'Image':
            st.markdown(" 1. Upload the Image and click Submit"   )
            st.markdown(" 2. The top 5 images will be displayed ") 

            uploaded_file = st.file_uploader("Upload image", type=[
                                                "png", "jpeg", "jpg"], 
                                                accept_multiple_files=False, 
                                                key=None, help="upload image")

            if uploaded_file is not None:
                user_input = Image.open(uploaded_file)
                st.image(user_input, caption='Uploaded Image.', 
                        width=200)
                st.write("")
        else:
            st.markdown(" Input the Text and click Submit"       )
            user_input = st.text_input('Enter your question here:')

        if st.button('Submit'):

            #Encode a text and image
            #Load CLIP model
            CATEGORY="IMAGES"
            pinecone.init(api_key=PINECONE_API_KEY, 
                          environment=PINECONE_REGION)
            index = pinecone.Index(PINECONE_INDEX_NAME)
            model= get_model_for_images()
            embedding = model.encode(user_input).tolist()
            xc = index.query(vector = embedding,
                                top_k=5,
                                filter={"category": {"$eq": CATEGORY}},
                                include_metadata=True)

            contexts = [
                (x['metadata'],x['score']) for x in xc['matches']
            ]

            for i, context in enumerate(contexts):
                st.write(f"Top {i+1} result: {context[0]['filename']} ,\
                         score : {context[1]}")
                image = Image.open(IMAGES_PATH + context[0]['filename'])
                st.image(image, 
                        caption='Uploaded Image.', 
                        width=200)
                st.write("")