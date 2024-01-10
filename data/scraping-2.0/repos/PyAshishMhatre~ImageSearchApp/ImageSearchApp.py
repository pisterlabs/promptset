import streamlit as st
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
import os
import openai
from urllib.request import urlopen
from io import BytesIO
import pinecone


openai.api_key = st.secrets['openai_api']



def GenSimilar(img, features, img_paths):  #Run search
    query = fe.extract(img)
    print('*********** Query **********', query.shape)
    print('*********** Features *********', features.shape)
    dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:10]  # Top 10 results
    scores = [(dists[id], img_paths[id]) for id in ids]
    
    col1, col2, col3 = st.columns(3)
    i = 0
    for score, image in scores:
        i = i + 1
        
        if i == 1:
            with col1:
                image = Image.open(image)
                st.image(image, caption='Score is {d_score}'.format(d_score = score))
        elif i == 2:
             with col2:
                image = Image.open(image)
                st.image(image, caption='Score is {d_score}'.format(d_score = score))
        else:
            with col3:
                image = Image.open(image)
                st.image(image, caption='Score is {d_score}'.format(d_score = score))
                i = 0     

def genimage(ask):
    try:
        response = openai.Image.create(
        prompt= ask,
        n=1,
        size="256x256"
        )
        image_url = response['data'][0]['url']
        st.image(image= image_url)
        
        image_file = urlopen(image_url)
        image_data = image_file.read()

        pil_image = Image.open(BytesIO(image_data))
        GenSimilar(pil_image, features, img_paths)
        
        
    except openai.error.OpenAIError as e:
        print(e.http_status)
        print(e.error)

@st.cache_resource
def intialize_pinecone():
    DATA_DIRECTORY = 'assignment4'
    INDEX_NAME = 'fashion'
    INDEX_DIMENSION = 4096
    BATCH_SIZE=200
    
    pinecone.init(api_key=st.secrets['pinecone_api'], environment=st.secrets['pinecone_env'])
    # if the index does not already exist, we create it
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, dimension=INDEX_DIMENSION)
    # instantiate connection to your Pinecone index
    index = pinecone.Index(INDEX_NAME)
    
    return index

@st.cache_data
def load_imgpath():
    root_dir = r'./static/img'
    # define dict
    files_path = {}
    
    #loop through the files 
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                #Extract img path
                img_path = os.path.join(subdir, file)
                #Extract subdict name and file name
                subdirectory_name = os.path.basename(subdir)
                file_name = os.path.splitext(file)[0] # e.g., ./static/img/xxx.jpg
                
                #append to dict 
                files_path['{sub}_{file}'.format(sub = subdirectory_name, file = file_name)] = img_path
    return files_path
                

def input_query(img,num,index):
    #Initialize feature extractor 
    feature = fe.extract(img).tolist()
    
    #query index
    response = index.query(
    feature, 
    top_k=num)
    return response

def output(response, files_path):
    #Read the response and display image
    col1, col2, col3 = st.columns(3)
    i = 0
    for responses in response['matches']:
        i = i + 1
        if i == 1:
            with col1:
                image = Image.open(  files_path['{path}'.format(path = responses['id'])]  ) 
                st.image(image, caption='Score is {d_score}'.format(d_score = responses['score']))
        elif i == 2:
            with col2:
                image = Image.open(  files_path['{path}'.format(path = responses['id'])]  ) 
                st.image(image, caption='Score is {d_score}'.format(d_score = responses['score']))
        else:
            with col3:
                image = Image.open(  files_path['{path}'.format(path = responses['id'])]  ) 
                st.image(image, caption='Score is {d_score}'.format(d_score = responses['score']))
                i = 0   

@st.cache_data   
def load_Feature_Img():
    root_dir = r'./static/img'
    
    features = []
    img_paths = []
    print('******  path  ****** ',os.walk(root_dir))
    #loop through the files 
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                #Extract img path
                img_path = os.path.join(subdir, file)
                #Appending in Image path list
                img_paths.append(img_path)
                
                subdirectory_name = os.path.basename(subdir)
                file_name = os.path.splitext(file)[0] 
                              
                features.append(np.load( Path("./static/feature") / (subdirectory_name + '_' + file_name + ".npy") ) )

    return features, img_paths
        
# Read image features
@st.cache_resource
def load_model():
    fe = FeatureExtractor()
    return fe

fe = load_model()
features, img_paths = load_Feature_Img()
features = np.array(features)

#Load the image from the path
image_path = "./banner.png"
banner = Image.open(image_path)
st.image(banner, use_column_width = True)

#Page Config
st.title('**:blue[Fashion Image Search]**')

#Take User Input
option = st.selectbox('How would you like to search?',('Upload an Image', 'Generate AI Image', 'Search using pinecone'))

if option == 'Upload an Image':
    file = st.file_uploader(label='Upload image to search', type=['jpg','png','jpeg'], key='FileInput')
    if file:
        st.image(file, caption='Uploaded image')
        img = Image.open(file)  # PIL image
        uploaded_img_path = r"./static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.name
        img.save(uploaded_img_path)
        GenSimilar(img, features, img_paths)
        
elif option == 'Generate AI Image':
    ask = st.text_input(label='Enter the description of clothing you want to customise (Eg:- Black batman logo top) ')
    run = st.button(label='Build', key='button1')  
    if run and ask != "":
        genimage(ask)  

elif option == 'Search using pinecone':
    index = intialize_pinecone()
    file_path = load_imgpath()
    file2 = st.file_uploader(label='Upload image to search', type=['jpg','png','jpeg'], key='FileInput2')
    num = st.number_input('Enter the number of images to match', min_value= 0, max_value=5, value=1, step=1)
    run = st.button(label='Search', key='button2') 
    if file2 and (num != 0) and run:
        st.image(file2, caption='Uploaded image')
        img = Image.open(file2)  # PIL image
        uploaded_img_path = r"./static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file2.name
        img.save(uploaded_img_path)
        
        response = input_query(img,num,index) 
        
        output(response, file_path)
        
else:
    st.markdown('Select search method !')







    

    
