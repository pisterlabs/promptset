import streamlit as st
import pandas as pd
import openai
import csv
import pandas
import os
from tqdm import tqdm
import re
import time
import random

def main():
    #st.image('./images/logo-removebg-preview.png')
    #"st.session_state object:",st.session_state

    if st.session_state.get("login_token") != True:
        st.error("You need login to access this page.")
        st.stop()

    # File upload section
    if 'categories_df' in st.session_state:

        st.subheader("uploaded categories:")
        st.write(st.session_state.get("categories_df"))
        up_again_pressed = st.button("upload again")
        if up_again_pressed:
            del st.session_state["categories_df"]
            cat_uploaded_file = st.file_uploader("Upload a file csv/txt", type=["csv","txt"],key="CatFile")

        # Display file content
            if cat_uploaded_file is not None:
                try:
                    choice = st.radio("Header present in file?", ("Yes", "No"))
                    if choice == 'Yes':
                        cat_df = pd.read_csv(cat_uploaded_file,header=0,names=["CATEGORY_NAME"])
                        st.text("File Contents:")
                        st.write(cat_df)
                    else:
                        cat_df = pd.read_csv(cat_uploaded_file,names=["CATEGORY_NAME"])
                        st.text("File Contents:")
                        st.write(cat_df)
                    ok_button_pressed = st.button("ok")
                    if ok_button_pressed:
                        st.session_state["categories_df"] = cat_df
                    #st.write(st.session_state.get("categories_df"))
                        st.success("Categories confirmed. Please navigate to classifAIr page to continue.")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif 'categories_df' not in st.session_state:

        cat_uploaded_file = st.file_uploader("Upload a file csv/txt", type=["csv","txt"],key="CatFile")


        # Display file content
        if cat_uploaded_file is not None:
            try:
                choice = st.radio("Header present in file?", ("Yes", "No"))
                if choice == 'Yes':
                    cat_df = pd.read_csv(cat_uploaded_file,header=0,names=["CATEGORY_NAME"])
                    st.text("File Contents:")
                    st.write(cat_df)
                else:
                    cat_df = pd.read_csv(cat_uploaded_file,names=["CATEGORY_NAME"])
                    st.text("File Contents:")
                    st.write(cat_df)
                ok_button_pressed = st.button("ok")
                if ok_button_pressed:
                    st.session_state["categories_df"] = cat_df
                    #st.write(st.session_state.get("categories_df"))
                    st.success("Categories confirmed. Please navigate to classifAIr page to continue.")
            except Exception as e:
                st.error(f"Error: {e}")




if __name__ == "__main__":
    main()

