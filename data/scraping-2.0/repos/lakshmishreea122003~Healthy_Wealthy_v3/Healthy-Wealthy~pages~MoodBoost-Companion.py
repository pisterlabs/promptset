import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pymongo

llm = OpenAI(temperature=0.9)
st.set_page_config(
    page_title=" MoodBoost Companion ",
    page_icon="🌞🌈",
)
st.markdown("<h1 style='color: #3B444B; font-style: italic; font-family: Comic Sans MS; font-size:4rem' >MoodBoost Companion</h1> <h3 style='color:#54626F; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Empowering Health and Happiness, One step at a time</h3>", unsafe_allow_html=True)

mongo_uri = "mongodb+srv://<username>:<password>@<cluster_url>/<database_name>?retryWrites=true&w=majority"

client = pymongo.MongoClient(mongo_uri)
db = client.get_database()
collection = db.get_collection("users")

name = st.text_input('Enter your name') 

result  = {}
if name is not "":
    query = {"name": name}
    result = collection.find_one(query)


name  = st.text_input("Enter your name here")

mood_template = PromptTemplate(
    input_variables = ['emotion'], 
    template='For {emotion} state of mind,in less than 15 words  give one good sentence  along with  a nice quote for the day to make me feel better.'
)
mood_chain = LLMChain(llm=llm, prompt=mood_template, output_key='mood')

if name is not "":
    res = mood_chain.run(result.get("mood"))
    st.write(res)
    
##########################
emotion_template = PromptTemplate(
    input_variables = ['emotion'], 
    template='For {emotion} state of mind,in less than 15 words  give one good sentence  along with  a nice quote for the day to make me feel better.'
)

 

emotion_chain = LLMChain(llm=llm, prompt=emotion_template, output_key='emotion')

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open("C:/Users/Lakshmi/Downloads/WebCam-Face-Emotion-Detection-Streamlit-main/WebCam-Face-Emotion-Detection-Streamlit-main/emotion_model1.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("C:/Users/Lakshmi/Downloads/WebCam-Face-Emotion-Detection-Streamlit-main/WebCam-Face-Emotion-Detection-Streamlit-main/emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier("C:/Users/Lakshmi/Downloads/WebCam-Face-Emotion-Detection-Streamlit-main/WebCam-Face-Emotion-Detection-Streamlit-main/haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                # llm_help = emotion_chain.run(output)
                # label = output + " " + llm_help
            
            label_position = (x, y)
            cv2.putText(img,output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #    
    st.markdown("<h2 style='color: green; font-style: italic; font-family: Comic Sans MS; ' > MoodBoost Companion 🌞🌈</h2> <h3 style='color: #ADFF2F; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Empowering Health and Happiness, One Step at a Time</h3>", unsafe_allow_html=True)

    st.header("Lets know your mood")
   
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
