import streamlit as st
import os

from langchain.document_loaders import PyPDFLoader
from langchain.llms import Ollama
from langchain.llms import OpenAI

import map_reduce
import refine

# Find what directory this file is located in
dir_path = os.path.dirname(os.path.realpath(__file__))

st.write("Example demo app on how to summarise a pdf")
st.markdown("""
Two summary methods demo'd here:
* Map-Reduce: Runs a map/reduce summary process. Each page is summarised (map phase) then a summary of summaries is
generated (reduce phase)
* Refine: Each doc (page) is summarised, then the summary and the next page are passed to the model to refine
the summary further, until there are no more pages. This will be slower than the Map-Reduce approach but may produce
better results.
""")

method = st.selectbox('Select Summary Method', ('Map-Reduce', 'Refine'), index=0)
summary = map_reduce.summarise if method == "Map-Reduce" else refine.summarise

# Make sure to run: `export OPENAI_API_KEY=...` if using openAI
llm = OpenAI() if os.getenv("OPEN_AI") else Ollama(model='mistral')

# Load PDF and split into pages
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf#using-pypdf
loader = PyPDFLoader(os.path.join(dir_path, 'goldilocks-story.pdf'))
docs = loader.load_and_split()

summary_container = st.empty()

# This can be VERY SLOW depending on the size of the PDF
with st.spinner(text=f"Generating {method} Summary...",  cache=True):
    # Use the model to generate a summary
    text = summary(llm, docs)
    with summary_container.container():
        st.markdown(text)



