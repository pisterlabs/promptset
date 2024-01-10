import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from constants import openai_api_key
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_api_key

st.title(": Language Translator : ")
llm=OpenAI(temperature=0.7)


language_name=st.sidebar.selectbox(
    label="select any Language",
    options=("English", "Mandarin Chinese", "Hindi", "Spanish", "French", 
             "Modern Standard Arabic", "Bengali", "Portuguese", "Urdu", 
             "Indonesian", "German", "Italian", "Turkish", "Vietnamese", 
             "Russian", "Thai", "Tamil", "Yue Chinese", "Marathi", 
             "Telugu", "Japanese", "Western Punjabi", "Wu Chinese", 
             "Korean", "French Creole", "Cantonese", "Malay", "Telugu", 
             "Urdu", "Gujarati", "Javanese", "Southern Pashto", "Burmese", 
             "Hakka Chinese", "Tagalog", "Ukrainian", "Yoruba", "Maithili", 
             "Uzbek", "Sindhi", "Amharic", "Farsi", "Yoruba", "Malayalam", 
             "Igbo", "Sundanese", "Dutch", "Kurdish", "Thai", "Egyptian Arabic", 
             "Filipino", "Kannada", "Moroccan Arabic", "Hausa", "Burmese", 
             "Polish", "Serbo-Croatian", "Nepali", "Sinhalese", "Kirundi", 
             "Zulu", "Czech", "Kinyarwanda", "Uyghur", "Swedish", 
             "Haitian Creole", "East Javanese", "Finnish", "Bhojpuri", 
             "Oromo", "Bulgarian", "Fula", "Malay", "Bambara", "Ilokano",
               "Hejazi Arabic", "Igbo", "Dinka", "Somali", "Latvian", 
               "Tajik", "Lithuanian", "Bashkir", "Kazakh", "Lao", "Lingala", 
               "Tatar", "Tswana", "Bavarian", "Low German", "Akan", "Aragonese",
                 "Batak Toba", "Bavarian", "Low German"),
    index=None
    )

st.write("Please choose the language  and then write the input text for translation")
sentence = st.chat_input("Say something")


input_prompt=PromptTemplate(
    input_variables=['sentence','language'],
    template="translate the following sentence '{sentence}' into {language}"
)

chain=LLMChain(
    llm=llm,
    prompt=input_prompt,
    output_key="translation",
    verbose=True
)


if language_name and sentence:
    st.write(chain({'sentence':sentence,'language':language_name}))
