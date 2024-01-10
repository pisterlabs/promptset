#! /usr/bin/env python

"""
@author: Ajay
Created on: 26/05/2023
Version: 0.0.1
GUIPandasAI is a simple Wrapper around PandasAI using Streamlit Framework - Helper for bundling it as EXE
"""
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import subprocess

if __name__ == '__main__':
    subprocess.run("streamlit run wrapper_streamlit.py")