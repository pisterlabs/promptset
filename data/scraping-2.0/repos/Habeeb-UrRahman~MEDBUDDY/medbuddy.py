import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os
import openai
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import pandas as pd

# Load the pre-computed features, disease names, and image file paths for Skin Disease Prediction
features_list = pickle.load(open("image_features_embedding_skin_disease.pkl", "rb"))
disease_names_list = pickle.load(open("disease_names_skin_disease.pkl", "rb"))
img_files_list = pickle.load(open("img_files_skin_disease.pkl", "rb"))

# Load the pre-trained ResNet model for Skin Disease Prediction
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Load models for Diabetes and Heart Disease Prediction
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Load the CSV file with image metadata for Ocular Disease Prediction
csv_file_path = r"C:\Users\rahma\OneDrive\Desktop\Eye Disease Prediction Model\full_df.csv"
df = pd.read_csv(csv_file_path)

# Set your OpenAI API key
openai.api_key = "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"

# Combine both Streamlit apps into one
st.title('WELCOME TO MEDBUDDY,')

# Sidebar for navigation
with st.sidebar:
    st.image('MEDBUDDY_logo.png', caption='MEDBUDDY', width=150)
    selected = option_menu('Select an App',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Skin Disease Prediction',
                           'Cardiovascular Risk Prediction',
                           'Ocular Disease Prediction',
                           'Med Buddy Bot'],
                           icons=['activity','heart','person','water','circle','robot'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Skin Disease Prediction Page
elif selected == 'Skin Disease Prediction':
    st.title('SKIN DISEASE PREDICTOR')
    
    def save_file(uploaded_file):
        try:
            with open(os.path.join("skin_disease_uploader", uploaded_file.name), 'wb') as f:
                f.write(uploaded_file.getbuffer())
                return 1
        except:
            return 0
    
    def extract_img_features(img_path, model):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expand_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img)
        result_to_resnet = model.predict(preprocessed_img)
        flatten_result = result_to_resnet.flatten()
        result_normalized = flatten_result / norm(flatten_result)
        return result_normalized
    
    def recommend(features, features_list, disease_names_list):
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(features_list)
        distance, indices = neighbors.kneighbors([features])
        return indices
    
    def recognize_disease(features, features_list, disease_names_list):
        indices = recommend(features, features_list, disease_names_list)
        recognized_disease = disease_names_list[indices[0][0]]
        return recognized_disease
    
    uploaded_file = st.file_uploader("Choose your image")
    
    if uploaded_file is not None:
        if save_file(uploaded_file):
            show_images = image.load_img(os.path.join("skin_disease_uploader", uploaded_file.name), target_size=(224, 224))
            st.image(show_images, caption="Uploaded Image", use_column_width=True)
            features = extract_img_features(os.path.join("skin_disease_uploader", uploaded_file.name), model)
            recognized_disease = recognize_disease(features, features_list, disease_names_list)
            st.subheader("Recognized Skin Disease:")
            st.write(recognized_disease)
            img_indices = recommend(features, features_list, disease_names_list)
            st.write("Top 5 similar skin disease images:")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i in range(5):
                with col1 if i % 5 == 0 else col2 if i % 5 == 1 else col3 if i % 5 == 2 else col4 if i % 5 == 3 else col5:
                    st.subheader(disease_names_list[img_indices[0][i]])
                    st.image(image.load_img(img_files_list[img_indices[0][i]], target_size=(224, 224)), caption=disease_names_list[img_indices[0][i]], use_column_width=True)
        else:
            st.error("Some error occurred while processing the image.")

# Ocular Disease Prediction Page
elif selected == 'Ocular Disease Prediction':
    st.title('Ocular Disease Prediction')
    st.write("This application supports retinal scan data for ocular disease prediction.")
    st.write("Upload a retinal scan image to find similar images and their corresponding diseases.")
    
    def save_file(uploaded_file):
        try:
            with open(os.path.join("ocular_disease_uploader", uploaded_file.name), 'wb') as f:
                f.write(uploaded_file.getbuffer())
                return 1
        except:
            return 0

    def extract_img_features(img_path, model):
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        expand_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img)
        result_to_resnet = model.predict(preprocessed_img)
        flatten_result = result_to_resnet.flatten()
        result_normalized = flatten_result / norm(flatten_result)
        return result_normalized

    def recommendd(features, features_list):
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(features_list)
        dist, indices = neighbors.kneighbors([features])
        return indices

    uploaded_file = st.file_uploader("Choose your retinal scan image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if save_file(uploaded_file):
            # Display image
            show_images = Image.open(uploaded_file)
            size = (500, 500)
            resized_im = show_images.resize(size)
            st.image(resized_im, caption='Uploaded Image', use_column_width=True)
            
            # Extract features of the uploaded image
            features = extract_img_features(os.path.join("ocular_disease_uploader", uploaded_file.name), model)
            
            # Find similar images
            img_indices = recommendd(features, features_list)
            
            st.subheader("Similar Images and Their Corresponding Diseases:")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            for i in range(5):
                img_index = img_indices[0][i]
                
                # Extract the filename (e.g., "0_left.jpg" or "0_right.jpg")
                filename = os.path.basename(img_files_list[img_index])
                
                # Determine whether it's a left or right image
                is_left = "_left" in filename
                is_right = "_right" in filename
                
                # Extract the diagnostic keywords based on the filename
                if is_left:
                    diagnostic_keywords = df[df["Left-Fundus"] == filename]["Left-Diagnostic Keywords"].values
                elif is_right:
                    diagnostic_keywords = df[df["Right-Fundus"] == filename]["Right-Diagnostic Keywords"].values
                else:
                    diagnostic_keywords = []
                
                col = [col1, col2, col3, col4, col5][i]
                
                # Display the diagnostic keywords (disease names) if available
                col.image(img_files_list[img_index], use_column_width=True, caption='Similar Image')
                if len(diagnostic_keywords) > 0:
                    col.write(f"Disease: {', '.join(diagnostic_keywords)}")
                else:
                    col.warning(f"Disease information not found in the CSV for {filename}")
        else:
            st.error("Some error occurred while processing the image.")


# OpenAI Chatbot Page
if selected == 'Med Buddy Bot':
    st.title('Med Buddy Bot')

    # Input for user query
    user_query = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        # Call the OpenAI API to get an answer
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can choose a different engine if needed
            prompt=user_query,
            max_tokens=50,  # Adjust the maximum response length as needed
        )

        # Display the answer to the user
        answer = response.choices[0].text
        st.write("Answer:", answer)



