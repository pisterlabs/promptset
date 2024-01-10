import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import joblib
import torch
import numpy as np
import cv2
import os

st.set_page_config(
    page_title="CattleCount Alert",
    page_icon="🌿",
)

st.markdown("<h1 style='color: green; font-style: italic; font-family: Comic Sans MS; font-size:5rem' >CattleCount Alert🐄</h1> <h3 style='color: #00755E; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Use Computer Vision for Automated Monitoring for Herd Size Management to prevent deforestation due to grazing</h3>", unsafe_allow_html=True)

def cattle_count(img):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(img)
    class_labels = results.names
    predictions = results.pred[0]  # Get predictions for the first image
    unique_class_labels = set(predictions[:, -1].tolist())
    detected_objects = [class_labels[int(label)] for label in unique_class_labels]
    st.image(np.squeeze(results.render()))

    # Count the number of sheep
    sheep_count = 0
    for box_info in predictions:
        class_index = int(box_info[-1])
        if class_labels[class_index] == 'sheep':
            sheep_count += 1

    return detected_objects, sheep_count

max = st.text_input("Enter the max count of cattle")
uploaded_file = st.file_uploader("Choose a file")


if uploaded_file and max is not None:
    st.write("image processing started")
    st.image(uploaded_file)
    img_array = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Save the uploaded file in the "pictures" folder
    pictures_folder = r"D:\llm projects\Forest-Amazon\pages\pictures"

    img_name = os.path.join(pictures_folder, "img.jpg")
    cv2.imwrite(img_name, img)

    object_list, count= cattle_count('D:/llm projects/EcoKids Hub/pictures/img.jpg')
    cattle = f'Number of cattle in the area is {count} and the max number allowed in the area is {max}'
    if count>int(max):
        st.markdown("<h1 style='color: red; font-style: italic; font-family: Comic Sans MS; font-size:5rem' >Too many cattles found.</h1> ", unsafe_allow_html=True)

        







