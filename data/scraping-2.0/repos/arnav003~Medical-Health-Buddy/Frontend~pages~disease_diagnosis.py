import streamlit as st
import pandas as pd
import pickle
from tensorflow import keras
import numpy as np
import openai

MAX_SYMPTOMS = 4
openai.api_key = st.secrets["OPENAI_API_KEY"]


def generate_response(messages, temperature=0.7, max_tokens=256, top_p=0.9, n=2, stop=None, frequency_penalty=0.9, presence_penalty=0.9):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    return response['choices'][0]['message']['content']


def chat():
    with st.expander("We have a few more questions..", expanded=True):
        if st.session_state.messages == []:
            st.session_state.messages = [
                {
                    "role": "user",
                    "content": f'''You are a doctor. 
                    A patient has the following symptoms: {" ".join(input_values)}.
                    You have to make a possible diagnosis. 
                    The primary diagnosis is {" ".join(st.session_state.predictions)}. 
                    Ask further questions to make the diagnosis more accurate. 
                    The questions must be framed in the way a doctor would ask a patient questions. 
                    If you need more information to make diagnosis, ask one question at a time. 
                    Only ask questions, and nothing else.
                    Ask questions and wait for the patient's answer.
                    Repeat this process until, you have all the information. 
                    If you have all the required information, then only return your final diagnosis.''',
                },
            ]
            response = generate_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st._rerun()

        for i, message in enumerate(st.session_state.messages[1:]):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(message["content"])

        user_input = st.text_area("You:", " ", key="user_input")
        submit_button = st.button("Submit")

        if submit_button and user_input.strip() != "":
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = generate_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st._rerun()


def show_prediction(input_values):
    st.session_state.predictions, other_predictions = create_input_df(input_values)

    st.write(f'''# Top Predicted Disease  
    {st.session_state.predictions[0]}  
    {st.session_state.predictions[1]}  
    {st.session_state.predictions[2]}  
    {st.session_state.predictions[3]}''')

    with st.expander("Predictions using other models:"):
        st.write(f'''
        LR: {other_predictions[0]}\n
        KNN: {other_predictions[1]}\n
        DT: {other_predictions[2]}''')

    st.session_state.messages = []
    st.session_state.prediction_made = True


def create_input_df(input_symptoms):
    predictions = []
    other_predictions = []
    input_symptoms_df = pd.DataFrame([input_symptoms_dict.values()], columns=symptoms_list)
    for symptom in input_symptoms:
        input_symptoms_df[symptom] = 1

    prediction_lr = model_lr.predict(input_symptoms_df)[0]
    other_predictions.append(prediction_lr)

    prediction_knn = model_knn.predict(input_symptoms_df)[0]
    other_predictions.append(prediction_knn)

    prediction_dt = model_dt.predict(input_symptoms_df)[0]
    other_predictions.append(prediction_dt)

    input_symptoms_array = np.array(input_symptoms_df, dtype=np.float32)
    prediction_dl = model_dl.predict(input_symptoms_array)[0]
    predictions.append(idx_to_disease[int(np.argpartition(prediction_dl, -1)[-1])] + ' (' + str(
        round(prediction_dl[np.argpartition(prediction_dl, -1)[-1]] * 100, 2)) + '%)')
    predictions.append(idx_to_disease[int(np.argpartition(prediction_dl, -2)[-2])] + ' (' + str(
        round(prediction_dl[np.argpartition(prediction_dl, -2)[-2]] * 100, 2)) + '%)')
    predictions.append(idx_to_disease[int(np.argpartition(prediction_dl, -3)[-3])] + ' (' + str(
        round(prediction_dl[np.argpartition(prediction_dl, -3)[-3]] * 100, 2)) + '%)')
    predictions.append(idx_to_disease[int(np.argpartition(prediction_dl, -4)[-4])] + ' (' + str(
        round(prediction_dl[np.argpartition(prediction_dl, -4)[-4]] * 100, 2)) + '%)')
    return predictions, other_predictions


@st.cache_resource
def get_model():
    pickle_in = open("Models/model_lr.pkl", "rb")
    model_lr = pickle.load(pickle_in)

    pickle_in = open("Models/model_knn.pkl", "rb")
    model_knn = pickle.load(pickle_in)

    pickle_in = open("Models/model_dt.pkl", "rb")
    model_dt = pickle.load(pickle_in)

    model_dl = keras.models.load_model("Models/model_dl.keras")

    return model_lr, model_knn, model_dt, model_dl


@st.cache_resource
def get_dataset():
    file = open('Dataset/symptoms.txt', 'r')
    symptoms = file.readlines()
    file.close()

    cleaned_symptoms_descriptions_df = pd.read_csv('Dataset/cleaned_symptoms_with_description.csv')
    cleaned_symptoms = cleaned_symptoms_descriptions_df["Cleaned Symptom"].tolist()
    symptoms_descriptions = cleaned_symptoms_descriptions_df["Description"].tolist()

    file = open('Dataset/diseases.txt', encoding='utf-8')
    diseases = file.readlines()
    file.close()

    diseases_descriptions_df = pd.read_csv('Dataset/disease_description.csv')
    diseases_descriptions = diseases_descriptions_df["Description"].tolist()

    return symptoms, cleaned_symptoms, symptoms_descriptions, diseases, diseases_descriptions


if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

symptoms, cleaned_symptoms, symptoms_descriptions, diseases, diseases_descriptions = get_dataset()

input_symptoms_dict = {}
symptom_to_idx = {}

for i, element in enumerate(symptoms):
    input_symptoms_dict[element.strip()] = 0
    symptom_to_idx[element.strip()] = i

symptoms_list = list(input_symptoms_dict.keys())

idx_to_disease = {}

for i, element in enumerate(diseases):
    idx_to_disease[i] = element.strip()

model_lr, model_knn, model_dt, model_dl = get_model()

st.markdown("# General Disease Diagnosis")

# camel_case_symptoms_list = []
# for string in symptoms_list:
#     camel_case_string = string[0].upper() + string[1:]
#     camel_case_symptoms_list.append(camel_case_string)

input_values = st.multiselect("Select symptoms:", symptoms_list, [], key="options")

if len(input_values) > MAX_SYMPTOMS:
    st.warning(f"Select up to {MAX_SYMPTOMS} symptoms. Please deselect some symptoms. Prediction might be inaccurate.")
    input_values = input_values[:MAX_SYMPTOMS]

if st.button("Get Prediction"):
    show_prediction(input_values)

if st.session_state.prediction_made:
    chat()
