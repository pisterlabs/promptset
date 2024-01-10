import streamlit as st
from PIL import Image
import io
import os
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI


os.environ["OPENAI_API_KEY"] = 'Your-OpenAI-key'

from transformers import pipeline

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

img=Image.open("C:\\Users\\OneDrive\\Pictures\\Camera Roll\\st.png")
st.set_page_config(page_title="Story.ai: Transform image into Narratives",page_icon=img)

st.title('Story:blue[.]ai')
tab1,tab2,tab3=st.tabs(['Home','Explore','Create'])


def img2text(url):
    #image_to_text=pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text=pipe(url)[0]["generated_text"]
    print(text)
    
    return text

def generate_story(scenario):
    template="""
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 100 words;
    
    CONTEXT: {scenario}
    STORY: 
    """
    prompt=PromptTemplate(template=template, input_variables=["scenario"])
    
    story_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=prompt,verbose=True)
    story=story_llm.predict(scenario=scenario)
    
    
    return story

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer hf_mpaIcXuxbxwvSJkqOzIPMnMabHeZV"}
    payloads = {
        
        "inputs":message
    }
    response=requests.post(API_URL,headers=headers,json=payloads)
    with open('input.mp3','wb') as file:
        file.write(response.content)

with tab1:

    st.write('Welcome to Story:blue[.]ai, where images transform into captivating audio tales. Unleash the power of AI storytelling – let your photos speak volumes. Every image has a story; let Story.ai bring yours to life with seamless narration. Turn your cherished moments into unforgettable stories, one image at a time.')
    img2=Image.open("C:\\Users\\Downloads\\every-picture-tells-a-story.jpg")
    img2=img2.resize((600, 300), Image.ANTIALIAS)
    st.image(img2)

    
    st.write('At Story:blue[.]ai, we harness the power of AI to turn your static images into dynamic narratives. Ever felt the story behind a photo? Now, let Story.ai bring it to life with vivid descriptions and engaging storytelling.')
    
    
    st.write('Here are some of the advantages of using Story:violet[.]ai:')
    
    st.success('''

 Effortless Transformation: No need to pen down your memories; let Story.ai effortlessly turn your images into audio stories.

Immersive Narration: Our AI doesn't just describe; it crafts narratives that transport you, adding a new dimension to your visual memories.

Accessibility: Make your stories accessible to all. Story.ai's audio format ensures everyone can experience the tales hidden in your images.

''')
    
with tab2:
    
        st.write("Every photo tells a story. Let Story.ai be the voice that shares your visual tales.")
        img3=Image.open("C:\\Users\\Downloads\\family.jpg")
        img3=img3.resize((800, 800), Image.ANTIALIAS)
        img4=Image.open("C:\\Users\\Downloads\\teaching.jpg")
        img4=img4.resize((800, 800), Image.ANTIALIAS)
        
        
        col1,col2=st.columns(2)
        with col1:
            st.image(img3,use_column_width=True)
            st.audio("family.mp3")
        with col2:
            st.image(img4,use_column_width=True)
            st.audio("teaching.mp3")
            
        img5=Image.open("C:\\Users\\Downloads\\dad.jpg")
        img5=img5.resize((800, 800), Image.ANTIALIAS)
        img6=Image.open("C:\\Users\\Downloads\\reading.jpg")
        img6=img6.resize((800, 800), Image.ANTIALIAS)
        
        col3,col4=st.columns(2)
        with col3:
            st.image(img5,use_column_width=True)
            st.audio("dad.mp3")
        with col4:
            st.image(img6,use_column_width=True)
            st.audio("reading.mp3")
        

        
with tab3:
    
    st.write("Experience the magic of storytelling like never before! Create captivating audio stories from your images effortlessly with the AI-powered tool, Story:blue[.]ai. Transform visuals into vivid narratives with a click.")
    uploaded_file = st.file_uploader("Choose an image", accept_multiple_files=False)  
    if uploaded_file:
        data=uploaded_file.read()
        image_buffer = io.BytesIO(data)
        pil_image = Image.open(image_buffer)
        converted_image = pil_image.convert('RGB')
        with st.columns(3)[1]:
            st.image(data,width=300)
        scenario=img2text(pil_image)
        story=generate_story(scenario)
        text2speech(story)
        st.audio("input.mp3")
        with st.expander("Story"):
            st.info(story)
