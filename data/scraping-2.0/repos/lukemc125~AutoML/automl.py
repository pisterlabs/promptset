from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling as yp
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import LabelEncoder
import pandasai
import os

secret_key = os.environ.get('MY_SECRET_KEY')
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token=secret_key)

pandas_ai = PandasAI(llm)

import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.title("Luke's Auto ML App")
    choice = st.radio("Navigation", ["Upload","Data Report","Chat","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = yp.ProfileReport(df)
    st_profile_report(profile_df)

if choice == "Chat": 
    st.title("Ask Anything About Your Data")
    prompt = st.text_area("Enter your prompt")

    if st.button('Run'):
        if prompt:
            with st.spinner("Generating response:"):
                st.write(pandas_ai.run(df, prompt=prompt))
        else:
            st.warning("Enter a prompt")


if choice == "Modelling": 
    chosen_target = st.selectbox('Choose Target Column', df.columns)

    for col in df.columns:
        if df[col].nunique() == 2:
            df[col] = df[col].astype('category').cat.codes
        
    if st.button('Run Modelling'): 
        try:
            setup_env = setup(df, target=chosen_target, numeric_imputation='median', feature_selection=True)
        except Exception as e:
            print(f"An error occurred during setup: {e}")
            # You can add additional code here to handle the error, if needed

        with st.spinner('Testing Models..'):
            setup_df = pull()
            # st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")