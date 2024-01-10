import os

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st

from llama_index import (
    Document,
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
)
from llama_index.llms import OpenAI


### Streamlit Multi-Page
st.set_page_config(
    page_title="Set Information",
    page_icon="📚",
)
