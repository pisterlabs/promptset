from langchain.llms import GooglePalm
import google.generativeai as palm
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) # read local .env file


llm = GooglePalm(temperature=0.2)
palm.configure(api_key=os.environ['GOOGLE_API_KEY_PALM'])