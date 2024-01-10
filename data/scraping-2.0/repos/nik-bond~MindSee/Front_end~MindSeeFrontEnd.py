import streamlit as st
from st_clickable_images import clickable_images
import requests
import numpy as np
import pandas as pd
import requests
import openai
from pathlib import Path
import base64
import json
import os
import PIL
import glob, random
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from google.cloud import storage

#  Path(__file__).resolve() get the current file path
# .parent goes back one so the folder of the current file
# folder / file_name e.g folder / "file_name" = "folder/file_name"

folder = Path(__file__).resolve().parent

st.markdown("""# MindSee
# Recreating images from MRI scans""")



#blob_path = "/subj01/test_images_indexed"
# @st.cache_data(persist=False)
def select_filter():
    option = st.selectbox('Select Images Filter',('all images', 'good prediction', 'interesting prediction'))
    return option

#@st.cache_data(persist=False)

#def image_filter(option):

option = select_filter()

if option == "all images":
    blob_path = "subj01/test_images_indexed"

if option == "good prediction":
    blob_path = "images/good"

if option == "interesting prediction":
    blob_path = "images/interesting"

    #return blob_path


# to load/upload images from Google Cloud buckets
@st.cache_data(persist=False)
def load_picfolder(blob_path):
    client = storage.Client()
    # access the bucket
    bucket = client.get_bucket('mindsee-preproc-data-2')
    # list all of the blobs
    images = list(bucket.list_blobs(prefix=blob_path))
    # shuffle the blobs
    random.shuffle(images)
    files = []
    for i in range(6):
        # take the first six blobs
        # get the name of file 0000.png
        file_name = images[i].id.split("/")[-2]
        # add the file name to the folder path
        # download to the file path
        image_bytes = images[i].download_as_bytes()
        image_bytes = base64.b64encode(image_bytes).decode()
        # add the file path to the list
        files.append((file_name, image_bytes))
    # return a list of six file paths where the new files stored
    return files

random_image_list = load_picfolder(blob_path)
# st.write(random_image_list)

# To generate random 5 images from the folder
# @st.cache_data(persist=False)
# def get_images():

#     random_image_lst= []
#     for i in range(1000):
#         #file_path_type = ["data/test_images_indexed/*.png"]
#         file_path_type = ["/Users/kansak/code/nik-bond/MindSee/data/test_images_indexed/*.png"]
#         images = glob.glob(random.choice(file_path_type))
#         random_image = random.choice(images)
#         # random_image_lst.append(random_image)
#         if random_image not in random_image_lst:
#             random_image_lst.append(random_image)
#             if len(random_image_lst)==6:
#                 break

#     return random_image_lst



# random_image_list = load_picfolder(blob_path)

# To get a picture grid

images = []
# for file in random_image_list:
#     with open(file, "rb") as image:
#         encoded = base64.b64encode(image.read()).decode()
#         print(encoded)
#         # st.markdown(images)
#         images.append(f"data:image/png;base64,{encoded}")
for file in random_image_list:
    images.append(f"data:image/png;base64,{file[-1]}")


clicked = clickable_images(
    images,
    titles=[f"Image #{str(i)}" for i in range(len(images))],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
)
st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")

if clicked == -1:
    uploaded_file = None
else:
    uploaded_file=random_image_list[clicked][0]

# st.markdown(uploaded_file)


if st.button("Generate New Images"):

    # Clear values from *all* all in-memory and on-disk data caches:
    # i.e. clear values from both square and cube
    st.cache_data.clear()
    streamlit_js_eval(js_expressions="parent.window.location.reload()")



if uploaded_file is None:
    st.markdown("Click an image")
else:
    st.write(f'<div><img src=data:image/png;base64,{random_image_list[clicked][1]} alt="image clicked" width=512 height=512></dive>', unsafe_allow_html=True)

# # #To show a brain scan pic
# brain_image = st.image(Image.open("images/brain_scan/7Re1.gif"))

if uploaded_file is None:
    response=None

else:
    # #Converting image name to string to send to API
    uploaded_file_name = uploaded_file

    uploaded_file_name = uploaded_file_name.rsplit('/', 1)[-1]

    uploaded_file_index = int(uploaded_file_name.rsplit(".png", 1)[0])
    # st.markdown(uploaded_file_index)
    # st.markdown(type(uploaded_file_index))
    # st.markdown(uploaded_file_name)



    # Get request to the API
    # Getting back the   predicted captions from the Model

    params= {'image_name':uploaded_file_index}
    response=requests.get(st.secrets["API_URL"], params=params).json()
    response = response[0]+', '+response[1]+' ,'+response[2]+' ,'+response[3] #+ ' photorealistic image'
    response_display = f"## {response}"
    st.markdown(response_display)




#Dall E
if st.button("Render Image"):
# if uploaded_file is not None:

    ## Creating the Dalle image as a json file
    PROMPT = response
    DATA_DIR = Path.cwd() / "responses"

    #creating directory for Dall E-json response
    DATA_DIR.mkdir(exist_ok=True)

    openai.api_key = st.secrets["OPENAI_KEY"]

    response = openai.Image.create(
        prompt=PROMPT,
        n=1,
        size="512x512",
        response_format="b64_json",
    )
    file_name = DATA_DIR / f"{PROMPT[:10]}-{response['created']}.json"

    with open(file_name, mode="w", encoding="utf-8") as file:
        json.dump(response, file)

    ##Converting and saving the image
    DATA_DIR = Path.cwd() / "responses"
    JSON_FILE = file_name
    IMAGE_DIR = Path.cwd() / "images" / JSON_FILE.stem

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_FILE, mode="r", encoding="utf-8") as file:
        response = json.load(file)

    for index, image_dict in enumerate(response["data"]):
        image_data = base64.b64decode(image_dict["b64_json"])
        image_file = IMAGE_DIR / f"{JSON_FILE.stem}-{index}.png"
        st.image(image_data)
        with open(image_file, mode="wb") as png:
            png.write(image_data)


# # Display the Dall E image on Streamlit
