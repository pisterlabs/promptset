import os
from pdfminer.high_level import extract_text
import openai
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st
import spacy

# Set your OpenAI API key here
api_key = 'your api key here'

# Load the Excel file with job descriptions
ds_jd_df = pd.read_excel(r"C:\Users\Yash\Desktop\DS-JD.xlsx")

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')

# Set your OpenAI API key here
api_key = 'your api key here'

# Load the Excel file with job descriptions
ds_jd_df = pd.read_excel(r"C:\Users\Yash\Desktop\DS-JD.xlsx")

# Download spaCy English language model
spacy.cli.download("en_core_web_sm")

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z0-9\s.]", "", text)

    # Tokenize the text using spaCy
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.is_alpha]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Join the cleaned words into a concise summary
    summary = " ".join(words)

    return summary

def clean_jd(ds_jd_df):
    jd_text = ds_jd_df['Job Description']
    jd_test = jd_text.apply(preprocess_text).str.cat(sep=' ')
    return jd_test

def analyze_resume(user_resume, job_description, api_key):
    openai.api_key = api_key

    prompt = f"You are a helpful assistant who understands the contents of a resume and a job description given by the user. Please analyze the user's resume and compare it to the job description. Focus on academic qualifications, skills, work experience, and projects. Highlight strengths and weaknesses for the specific job role and suggest skills to improve:\n\nUser Resume:\n{user_resume}\n\nJob Description:\n{job_description}\n\nStrengths:\n- Compare the skills, projects, and degrees mentioned in the user's resume to those in the job description and highlight matching skills/degrees as strengths.\n- Assess the relevance of the user's work experience to the job description.\n- Highlight notable projects completed by the user, especially if they are relevant to the job description.\n\nWeaknesses:\n- Identify any differences between the user's degree and the degree mentioned in the job description.\n- Suggest areas of improvement for the user's projects, such as [mention improvements].\n\nSuggested Improvements:\n- Recommend ways for the user to enhance skills in [mention skills], such as taking courses, earning certifications, or working on relevant projects.\n- Suggest opportunities for the user to gain more experience in [mention areas].\n\nOverall Assessment:\n- Provide an overall assessment of the user's suitability for the job.\n- If there are substantial differences between the job description and the user's resume, indicate that the user's resume may not fully align with the job.\n\nPlease provide detailed insights and suggestions."

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=800
    )

    job_insights = response.choices[0].text.strip()

    return job_insights

st.title("Resume Analyzer")
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf"])

if uploaded_file is not None:
    st.success("Resume uploaded successfully")

    jd_test = clean_jd(ds_jd_df)

    user_jd = st.text_input("Copy and paste your custom job description here:")

    if st.button("Custom Analysis"):
        cleaned_resume = preprocess_text(extract_text(uploaded_file))
        custom_analysis = analyze_resume(cleaned_resume, user_jd, api_key)
        st.write(custom_analysis)

    if st.button("Analyse Resume"):
        cleaned_resume = preprocess_text(extract_text(uploaded_file))
        analysed_resume = analyze_resume(cleaned_resume, jd_test, api_key)
        st.write(analysed_resume)


# streamlit run D:\VSCodium\Guvi-Projects\Resume_analyzer.py