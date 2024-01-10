from openai import OpenAI
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
#load_dotenv("../../.env", override=True)
load_dotenv("../.env", override=True)

os.environ.get("OPENAI_API_KEY")
