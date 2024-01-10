#! /usr/bin/env python

"""
@author: Ajay
Created on: 26/05/2023
Version: 0.0.1
`GUIPandasAI` is a simple python wrapper around PandasAI using the Streamlit Framework with key data analysis functionalities
"""
import streamlit as st
import pandas as pd
from IPython.display import display
import numpy as np
import lux
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import scipy.stats as stats
import datapane as dp

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

def generate_insights_one(df):
    insights = []

    # Summary statistics
    insights.append("Summary Statistics:")
    insights.append(df.describe().to_string()) #

    # Missing values
    missing_values = df.isnull().sum()
    insights.append("Missing Values:")
    insights.append(missing_values.to_string())
    
    # Missing values summary
    missing_values_count = df.isnull().sum()
    missing_values_percent = (missing_values_count / len(df)) * 100
    missing_values_summary = pd.DataFrame({
        "Missing Values": missing_values_count,
        "Missing Values %": missing_values_percent
    })
    insights.append("Missing Values Summary:\n")
    insights.append(missing_values_summary.to_string())
    
    # Data types
    data_types = df.dtypes
    insights.append("Data Types:")
    insights.append(data_types.to_string())

    # Unique values
    unique_values = df.nunique()
    insights.append("Unique Values:")
    insights.append(unique_values.to_string())

    # Correlation matrix
    correlation_matrix = df.corr()
    insights.append("Correlation Matrix:")
    insights.append(correlation_matrix.to_string())

    return "\n\n".join(insights)

def generate_trends_and_patterns_one(df):
    trends_and_patterns = []

    # Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], ax=ax)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        trends_and_patterns.append(fig)

    # Pairwise scatter plots
    sns.set(style="ticks")
    scatter_matrix = sns.pairplot(df, diag_kind="kde")
    trends_and_patterns.append(scatter_matrix.fig)

    return trends_and_patterns

def generate_insights(df):
    insights = []
    
    # Summary statistics
    summary_stats = df.describe()
    insights.append("Summary Statistics:\n" + summary_stats.to_string())

    # Missing values
    missing_values_count = df.isnull().sum()
    missing_values_percent = (missing_values_count / len(df)) * 100
    missing_values_summary = pd.DataFrame({
        "Missing Values": missing_values_count,
        "Missing Values %": missing_values_percent
    })
    insights.append("Missing Values Summary:\n" + missing_values_summary.to_string())

    # Correlation matrix
    correlation_matrix = df.corr()
    insights.append("Correlation Matrix:\n" + correlation_matrix.to_string())

    return "\n\n".join(insights)

def generate_trends_and_patterns(df):
    trends_and_patterns = []

    # Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], ax=ax)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title("Distribution of " + col)
        plt.tight_layout()
        trends_and_patterns.append(fig)

    return trends_and_patterns

def aggregate_data(df, columns):
    aggregated_data = df.groupby(columns).size().reset_index(name='Count')
    return aggregated_data

def generate_profile_report(df):
    profile = ProfileReport(df, explorative=True)
    # To Generate a HTML report file
    profile.to_file("profiling_results.html")
    return profile   

def render_sql_view(df):
    view = dp.Blocks(dp.DataTable(df))
    return dp.save_report(view, path="SQL_Rendered_View.html", open=True)

if check_password():
    st.title("A Simple GUI-based APP for making DataFrames Coversational with key data analysis utilities!!!")
    fig = None
    #response_history = []
    response_history = st.session_state.get("response_history", [])
    if "openai_key" not in st.session_state:
        with st.form("API key"):
            key = st.text_input("OpenAI Key", value="", type="password")
            if st.form_submit_button("Submit"):
                st.session_state.openai_key = key
                st.session_state.prompt_history = []
                st.session_state.df = None
    
    if "openai_key" in st.session_state:
        st.write(
            "Looking for an example *.csv-file?, check [here](https://github.com/ajayarunachalam/pynmsnn/blob/main/data/iris_data.csv)."
        )
        if st.session_state.df is None:
            uploaded_file = st.file_uploader(
                "Choose a Single CSV File..",
                type="csv",
            )
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
        if st.session_state.df is not None:
            st.subheader("Peek into the uploaded dataframe:")
            st.write(st.session_state.df.head(2))

        with st.form("Question"):
            question = st.text_area("Question", value="", help="Enter your queries here")
            #answer = st.text_area("Answer", value="")
            submitted = st.form_submit_button("Submit")
            if submitted:
                with st.spinner():
                    llm = OpenAI(api_token=st.session_state.openai_key)
                    pandas_ai = PandasAI(llm)
                    x = pandas_ai.run(st.session_state.df, prompt=question)
                    
                    if "insights" in question.lower():
                        insights = generate_insights_one(st.session_state.df)
                        st.write(insights)
                    elif "trends" in question.lower() or "patterns" in question.lower():
                        trends_and_patterns = generate_trends_and_patterns_one(st.session_state.df)
                        for fig in trends_and_patterns:
                            st.pyplot(fig)
                    elif "aggregate" in question.lower():
                        columns = question.lower().split("aggregate ")[1].split(" and ")
                        aggregated_data = aggregate_data(st.session_state.df, columns)
                        st.subheader("Aggregated Data:")
                        st.write(aggregated_data)
                    elif "profile" in question.lower():
                        profile = generate_profile_report(st.session_state.df)
                        if profile:
                            st.write("Check Profile Report in root directory")
                    elif "sql" in question.lower() or "SQL" in question.lower() or "view" in question.lower() or "VIEW" in question.lower():
                        render_sql_view(st.session_state.df)
                        st.write("SQL View Rendered.. Check 'SQL_Rendered_View.html' file")
                                     
                    fig = plt.gcf()
                    #fig, ax = plt.subplots(figsize=(10, 6))
                    plt.tight_layout()
                    if fig.get_axes() and fig is not None:
                        st.pyplot(fig)
                        fig.savefig("plot.png")
                    st.write(x)
                    st.session_state.prompt_history.append(question)
                    response_history.append(x)  # Append the response to the list
                    st.session_state.response_history = response_history
        

        st.subheader("Prompt history:")
        st.write(st.session_state.prompt_history)
        
        st.subheader("Prompt response:")
        for response in response_history:
            st.write(response)
 
    if st.button("Clear"):
        st.session_state.prompt_history = []
        st.session_state.response_history = []
        st.session_state.df = None
        
    if st.button("Save Results", key=0):
        with open("historical_data.txt", "w") as f:
            for response in response_history:
                f.write(response + "\n")
        if fig is not None:
            fig.savefig("plot.png")  
        
    st.write('---')
    st.text('')
    st.markdown(
            '`Created by` [Ajay](https://www.linkedin.com/in/ajay-ph-d-4744581a/) | \
             `Github:` [GitHub](https://github.com/ajayarunachalam/)')