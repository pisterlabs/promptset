import streamlit as st
from PIL import Image
import cv2
import numpy as np
import easyocr
import glob
from math import sqrt
import openai
import os
import pandas as pd

def profanity_check():

    with open("prompt_one.txt",'r') as f:
        prompt_one = f.read()
            
    st_list1 = []
    all_imgs = []
    Preview_button = False
    Check_button = False
    file_uploaded = []
    text_list = []
    response_list = []

    # To extact text from the Licence plate

    def text_extraction(all_imgs):
        confidence_threshold = 0.2
        reader = easyocr.Reader(['en'])
        for image in all_imgs:
            var = " "
            image = cv2.resize(image,(500,300))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            max = 0
            ocr_results = reader.readtext(image)
            for detection in ocr_results:
                if detection[2] > confidence_threshold:
                    x0,y0 = [int(value) for value in detection[0][0]]
                    x1,y1 = [int(value) for value in detection[0][1]]
                    x2,y2 = [int(value) for value in detection[0][2]]
                    x3,y3 = [int(value) for value in detection[0][3]]
                    dist = (sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
                    if dist > max:
                        max = dist
                        var = detection[1]
            text_list.append(var)
        return text_list
    
    # GPT 3 code to produce response for the licence plate text 

    def word_check(text):
        openai.api_key = os.environ['API_KEY']
        prompt = prompt_one + text
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.82,
        max_tokens=1200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
        return (response["choices"][0].text) 
        
    # Building streamlit application 

    
    c11,c12,c13,c14,c15 = st.columns([0.25,1.5,2.75,0.25,1.75])

    with c12:
        # st.write("")
        st.write("")
        st.write("")
        st.markdown("#### **Problem Statement**")
    with c13:
        pb_select = st.selectbox("",['Select the problem Statement','Check for Profanity on Licence Plate'],key = "key7")
    with c11:
        st.write("")
    with c14:
        st.write("")
    with c15:
        st.write("")

    st_list1 = ['Check for Profanity on Licence Plate']

    with c11:
        st.write("")
    with c12:
        if pb_select  in st_list1:
            st.write("")
            st.write("")
            st.write("")
            st.markdown("#### **Upload Input Data**")
    with c13:
        if pb_select in st_list1:
            file_uploaded = st.file_uploader("Choose a image file", type=["JPG",'JFIF','JPEG','PNG','TIFF',],accept_multiple_files=True)
            if file_uploaded is not None:
                for file in file_uploaded:

                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    all_imgs.append(cv2.imdecode(file_bytes, 1))
    with c14:
        st.write("")
    with c15:
        if pb_select  in st_list1 and len(all_imgs)!=0:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            Preview_button = st.button('Preview')

    # To Display images when preview button is pressed

    if Preview_button is True and len(all_imgs)!=0:
        cd1,cd2,cd3,cd4,cd5 = st.columns((2,2,2,2,2))
        if len(all_imgs) >=5:
            Display_Images= all_imgs[0:5]
            for i in range(len(Display_Images)):
                with cd1:
                    st.image(all_imgs[i])
                with cd2:
                    st.image(all_imgs[i+1])
                with cd3:
                    st.image(all_imgs[i+2])
                with cd4:
                    st.image(all_imgs[i+3])
                with cd5:
                    st.image(all_imgs[i+4])
                    break
        else:
            with c11:
                st.write("")
            with c12:
                st.write("")
            with c13:
                st.error("Upload atleast 5 Images to preview")
            with c14:
                st.write("")
            with c15:
                st.write("")

    with c12:
        st.write("")
        st.write("")
    with c13:
        if pb_select  in st_list1 and len(all_imgs)!=0:
            Check_button = st.button("Check for Profanity")
            if Check_button == True:
                text_list = text_extraction(all_imgs)
                for text in text_list:
                    st.success(text)
                    try:
                        response = word_check(text) # Sending text to GPT3 for response
                        response = response.replace("\n","") # Removing \n from response
                        sp = response.split("|") # Apply split with respect to |
                        if  '' in sp: # Removing empty spaces
                            sp.remove('')
                        if len(sp)>=21: # To display a table(dataframe) we should atleast have 21 elements(strings) 7 rows * 3 Columns
                            category = []
                            probability = []
                            reason = []
                            i = 0
                            while i<len(sp):
                                category.append(sp[i])
                                probability.append(sp[i+1])
                                reason.append(sp[i+2])
                                i += 3

                            dictionary = dict({"Category":category,"Probability":probability,"Reason":reason})
                            import pandas as pd
                            dataframe = pd.DataFrame(dictionary)
                            dataframe = dataframe.iloc[1:,:]
                            st.dataframe(dataframe,width=700,height=300)
                        else:
                            continue
                    except Exception as e:
                            st.markdown(e)
 
    with c11:
        st.write("")
    with c14:
        st.write("")
    with c15:
        st.write("")

