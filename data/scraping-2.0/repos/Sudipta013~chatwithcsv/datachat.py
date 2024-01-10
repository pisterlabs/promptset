import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import pandas_bokeh
import missingno

from dotenv import load_dotenv
import os
import tempfile
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
#from sqlalchemy.ext.declarative import declarative_base

def create_correlation_chart(corr_df): ## Create Correlation Chart using Matplotlib
    fig = plt.figure(figsize=(15,15))
    plt.imshow(corr_df.values, cmap="Blues")
    plt.xticks(range(corr_df.shape[0]), corr_df.columns, rotation=90, fontsize=15)
    plt.yticks(range(corr_df.shape[0]), corr_df.columns, fontsize=15)
    plt.colorbar()

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[0]):
            plt.text(i,j, "{:.2f}".format(corr_df.values[i, j]), color="red", ha="center", fontsize=14, fontweight="bold")

    return fig

def create_missing_values_bar(df):
    missing_fig = plt.figure(figsize=(10,5))
    ax = missing_fig.add_subplot(111)
    missingno.bar(df, figsize=(10,5), fontsize=12, ax=ax)

    return missing_fig

def find_cat_cont_columns(df): ## Logic to Separate Continuous & Categorical Columns
    cont_columns, cat_columns = [],[]
    for col in df.columns:        
        if len(df[col].unique()) <= 25 or df[col].dtype == np.object_: ## If less than 25 unique values
            cat_columns.append(col.strip())
        else:
            cont_columns.append(col.strip())
    return cont_columns, cat_columns

### Web App / Dashboard Code
st.set_page_config(page_icon=":bar_chart:", page_title="EDA Automated using Python")
open_ai_key = st.secrets["auth_token"]
# CSS
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

bg = """
        <style> [data-testid="stAppViewContainer"]
        {
            background: rgb(33, 36, 38);
        }
        </style>
        """
st.markdown(bg, unsafe_allow_html=True)
# Add the customized yellow bottom bar on the left side
bottom_bar_html = """
    <style>
    .bottom-bar {
        background-color: #FFA500;
        padding: 10px;
        position: fixed;
        left: 0;
        bottom: 0;
        height: 100%;
        width: 70px;
        writing-mode: vertical-rl; /* Vertical writing mode */
        text-orientation: upright; /* Text orientation */
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    <div class="bottom-bar">
        <span style="color: Black; font-family: 'Russo One'; font-size: 18px;">THE TECHIE INDIANS</span>
    </div>
    """
st.markdown(bottom_bar_html, unsafe_allow_html=True)

load_dotenv()
#st.title("Harnessing the Power of Python to Automate EDA :bar_chart: :tea: :coffee:")
st.markdown("<h1 style='text-align: center; font-family:Abril Fatface ; -webkit-text-stroke: 1px yellow ;font-size: 60px; padding-bottom: 15px; color: rgb(255, 255, 255) ;'>Ask Your CSV</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; font-size: 15px; color: rgba(255,255,255,0.5);'>Upload CSV file to see various Charts related to EDA. Please upload file that has both continuous columns and categorical columns.</h5>", unsafe_allow_html=True)
upload = st.file_uploader(label="Upload File Here:", type=["csv"])

if upload is not None: ## File as Bytes
    #df = pd.read_csv(upload)

    tab0, tab1, tab2, tab3 = st.tabs(["Chat with CSV","Dataset Overview :clipboard:", "Individual Column Stats :bar_chart:", "Explore Relation Between Features :chart:"])

    with tab0:
        if upload is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(upload.read())
                csv_file_path = temp_file.name
            #read and display dataset
            st.subheader("CSV dataset")
            df = pd.read_csv(csv_file_path)
            st.dataframe(df)
            #chatcsv
            st.info("Chat with your csv")
            input_text = st.text_input("Enter your Query")
            if input_text is not None:
                if st.button("chat"):
                    agent = create_csv_agent(OpenAI(temperature=0,openai_api_key = open_ai_key),csv_file_path)
                    with st.spinner(text="In progress..."):
                        st.info("Your query: " + input_text)
                        st.write(agent.run(input_text))
    with tab1: ## Dataset Overview Tab        
        st.subheader("1. Dataset")
        st.write(df)

        cont_columns, cat_columns = find_cat_cont_columns(df)

        st.subheader("2. Dataset Overview")
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Rows", df.shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Duplicates", df.shape[0] - df.drop_duplicates().shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Features", df.shape[1]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Categorical Columns", len(cat_columns)), unsafe_allow_html=True)
        st.write(cat_columns)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Continuous Columns", len(cont_columns)), unsafe_allow_html=True)
        st.write(cont_columns)
        
        corr_df = df[cont_columns].corr()
        corr_fig = create_correlation_chart(corr_df)
        
        st.subheader("3. Correlation Chart")
        st.pyplot(corr_fig)

        st.subheader("4. Missing Values Distribution")
        missing_fig = create_missing_values_bar(df)
        st.pyplot(missing_fig)

    with tab2: ## Individual Column Stats
        df_descr = df.describe()
        st.subheader("Analyze Individual Feature Distribution")

        st.markdown("#### 1. Understand Continuous Feature")        
        feature = st.selectbox(label="Select Continuous Feature", options=cont_columns, index=0)

        na_cnt = df[feature].isna().sum()
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Count", df_descr[feature]['count']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {} / ({:.2f} %)".format("Missing Count", na_cnt, na_cnt/df.shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {:.2f}".format("Mean", df_descr[feature]['mean']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {:.2f}".format("Standard Deviation", df_descr[feature]['std']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Minimum", df_descr[feature]['min']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Maximum", df_descr[feature]['max']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> :".format("Quantiles"), unsafe_allow_html=True)
        st.write(df_descr[[feature]].T[['25%', "50%", "75%"]])
        ## Histogram
        hist_fig = df.plot_bokeh.hist(y=feature, bins=50, legend=False, vertical_xlabel=True, show_figure=False)
        st.bokeh_chart(hist_fig, use_container_width=True)

        st.markdown("#### 2. Understand Categorical Feature")
        feature = st.selectbox(label="Select Categorical Feature", options=cat_columns, index=0)
        ### Categorical Columns Distribution        
        cnts = Counter(df[feature].values)
        df_cnts = pd.DataFrame({"Type": cnts.keys(), "Values": cnts.values()})
        bar_fig = df_cnts.plot_bokeh.bar(x="Type", y="Values", color="tomato", legend=False, show_figure=False)
        st.bokeh_chart(bar_fig, use_container_width=True)

    with tab3: ## Explore Relation Between Features
        st.subheader("Explore Relationship Between Features of Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(label="X-Axis", options=cont_columns, index=0)
        with col2:
            y_axis = st.selectbox(label="Y-Axis", options=cont_columns, index=1)

        color_encode = st.selectbox(label="Color-Encode", options=[None,] + cat_columns)

        scatter_fig = df.plot_bokeh.scatter(x=x_axis, y=y_axis, category=color_encode if color_encode else None, 
                                    title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()),
                                    figsize=(600,500), fontsize_title=20, fontsize_label=15,
                                    show_figure=False)
        st.bokeh_chart(scatter_fig, use_container_width=True)
