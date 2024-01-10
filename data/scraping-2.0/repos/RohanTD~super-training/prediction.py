import streamlit as st
import pandas as pd
import numpy as np
import openai
import pandas as pd
import numpy as np
import joblib
import sklearn
import pickle
import util
st.title("Imaging Predictor")

model = joblib.load("final_svm_probability.pkl")
pkl_file = open("encodings.pkl", "rb")
lbl = pickle.load(pkl_file)
pkl_file.close()

input_text = st.text_input("Enter the clinical observations of the patient:")


def get_embedding(text, model="text-embedding-ada-002"):
    print(text)
    text = text.replace("\n", " ")
    st.write(openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"])
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

if st.button("Submit"):
    vals = util.process_text(input_text)
    input_text = " ".join([*set(vals)])

    input_embedded = np.array(get_embedding(input_text, model="text-embedding-ada-002"))
    st.write(input_embedded)
    
    input_embedded = input_embedded.reshape(1, -1)
    st.write(input_embedded)
    y_pred_proba = model.predict_proba(input_embedded)
    st.write(y_pred_proba)
    y_pred = model.predict(input_embedded)
    st.write(y_pred_proba)

    print(y_pred_proba)
    st.write(
        str(lbl.inverse_transform(y_pred)[0])
        + ": {:.2f}%".format(float(str(y_pred_proba[0][y_pred])[1:-1]) * 100)
    )
