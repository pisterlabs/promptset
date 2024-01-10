import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import requests
from OpenAIUtils import query_to_summaries, questions_to_answers 
from EDGARFilingUtils import (
    get_all_submission_ids, 
    get_text_from_files_for_submission_id, 
    split_text, 
    filter_chunks, 
    ROOT_DATA_DIR,
    TICKER_TO_COMPANY_NAME,
    QUESTION_TO_CATEGORY
)
import openai
openai.api_key = st.secrets["openai_api_key"]

from transformers import GPT2TokenizerFast
# Used to check the length of the prompt.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

st.set_page_config(layout="wide")

### Streamlit app starts here
st.title("Play with GPT-3 Completion API and 10-Ks")

list_of_questions = list(QUESTION_TO_CATEGORY.keys())
#Change this question to add to the UI
question_to_add = "What are the climate opportunities this company faces?"
list_of_questions.append(question_to_add)
relevant_questions = st.multiselect("Select questions to use for search within the text.",
                                    list_of_questions,default=[question_to_add])
re_embed = not st.checkbox("Re-calculate Document Embeddings")
temperature = st.number_input("GPT-3 Temperature",min_value = 0., max_value = 1., value=0.5, step=0.05)
if st.button("Search for relevant sections to list of questions"):
    df_completions = query_to_summaries(relevant_questions,temperature,print_responses=False)
    st.table(df_completions)