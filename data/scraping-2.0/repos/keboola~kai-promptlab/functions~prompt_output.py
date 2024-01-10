# prompt_output.py

import streamlit as st
import pandas as pd
import re
import random

from functions.get_parameters import get_parameters
from functions.chat_completion import openai_api_call
from functions.processing_messages import messages

# Get user's prompts and params
def get_prompts(num_prompts):
    prompts_dict = {}
    columns = st.columns(num_prompts)
    for i in range(num_prompts):
        with columns[i]:
            prompt_input = st.text_area(
                f'Prompt {i + 1}:', 
                placeholder=f'Prompt {i + 1}:', 
                label_visibility="collapsed", height=300,
                key=f"prompt_{i + 1}"
            )
            prompts_dict[f"prompt_{i + 1}"] = prompt_input

            with st.expander(f"__Prompt {i + 1} parameter settings__"):
                response_params = get_parameters(f"prompt_{i + 1}")
                st.session_state[f"response_params_{i + 1}"] = response_params
    return prompts_dict

# Create placeholder list
def find_placeholders(prompts):
    placeholder_set = set()
    for prompt_key in prompts.keys():
        if prompts[prompt_key]:
            matches = re.findall(r'\[\[(.*?)\]\]', prompts[prompt_key])
            placeholder_set.update(matches) 

    placeholder_list = list(placeholder_set)
    return placeholder_list


# Get relevant cols to output table
def add_relevant_cols(df, prompts, placeholder_list):
    prompt_output = pd.DataFrame(index=df.index)

    for prompt_key in prompts.keys():
        if prompts[prompt_key]:
            relevant_cols = [col for col in placeholder_list if col in df.columns]
            for col in relevant_cols:
                prompt_output[col] = df[col]
    return prompt_output

# Get responses 
def create_reponse_df(df, prompts, placeholder_list, prompt_output):
    for idx, prompt_key in enumerate(prompts.keys()):
        random_message = random.choice(messages)
        if prompts[prompt_key]:
            apply_prompt_state = st.text(random_message)
            placeholder_columns = placeholder_list
            result_series = df.apply(
                openai_api_call, 
                args=(prompts[prompt_key], prompt_key, placeholder_columns, st.session_state[f"response_params_{idx+1}"]),
                axis=1
            )
            prompt_output[prompt_key] = result_series[prompt_key]
            apply_prompt_state.text("Done!")
    return prompt_output

def get_response(df, prompts):
    placeholder_list = find_placeholders(prompts)
    prompt_output = add_relevant_cols(df, prompts, placeholder_list)
    prompt_output = create_reponse_df(df, prompts, placeholder_list, prompt_output)
    return prompt_output
