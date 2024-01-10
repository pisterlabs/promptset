import streamlit as st
import openai
import os
import re
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
from plugins.llms import get_llm_output



#----------FUNCTIONS TO GENERATE PROMPTS----------------

def col_names_and_types(df):
    """Given a df, it returns a string with the column names and types"""
    col_names = df.columns
    col_types = df.dtypes
    col_names_and_types = []
    for col_name, col_type in zip(col_names, col_types):
        col_names_and_types.append(f"{col_name}: {col_type}")
    col_names = '\n- '.join(col_names_and_types)
    buf = f"""The Dataframe has the following columns: 
    
    - {col_names}"""
    return buf

def get_sample_data_as_markdown(df):
    """
    Gets a sample of 5 lines, if there are at least 3 lines.
    Else, gets the whole thing.
    """
    if len(df) > 5:
        buf = df.head(5).to_markdown(index=False)
    else:
        buf = df.to_markdown(index=False)
    return buf

def get_function_prompt(df, default=None):
    """Returns a prompt for GPT-3"""
    
    prompt_for_table = ''
    
    if st.sidebar.checkbox(f"Show sample data", value=False):
        st.dataframe(df.sample(5))
    prompt_for_table += col_names_and_types(df) + '\n'
    buf = get_sample_data_as_markdown(df)
    prompt_for_table += f"\nHERE IS SOME SAMPLE DATA:\n{buf}\n"
    if not default:
        default = ""
    label = 'I want to'
    if 'i_want_to' in st.session_state:
        if st.session_state.i_want_to:
            label = st.session_state.i_want_to
        
    i_want_to = st.text_area(label, value=default)

    # Check if there is a the_func in the session state
    # with the name my_func.  If yes, add this as context.
    func_prompt = ''
    if 'the_func' in st.session_state:
        if 'my_func' in st.session_state.the_func:
            func_prompt = f"FYI: This is the function I am trying to modify: {st.session_state.the_func}"

    prompt = f"""
    I am working on a dataframe named df.  

    {prompt_for_table}

    {func_prompt}
    
    I WANT TO: {i_want_to}

    """

    return prompt


def clean_function(the_func):
    """
    Cleans the function in a number of ways.
    """
    # Remove the first line if just mentions the language
    if the_func.startswith('python'):
        the_func = '\n'.join(the_func.split('\n')[1:])
    
    # Remove import statements
    revised_func = ''
    for line in the_func.split('\n'):
        if  line.startswith('import'):
            pass
        # If it's a function def, add it
        elif line.startswith('def'):
            revised_func += line + '\n'
        # If it's calling my_func, pass
        elif 'my_func' in line:
            pass
        else:
            revised_func += line + '\n'

    st.session_state.the_func = revised_func
    return revised_func

def parse_code_or_requirements_from_response(response):
    """
    The LLM can return code or requirements in the content.  
    Ideally, requirements come in triple pipe delimiters, 
    but sometimes they come in triple backticks.

    Figure out which one it is and return the extracted code or requirements.
    """
    # If there are ```, it could be code or requirements
    code_or_requirement = None
    if '```' in response:
        # If it's python code, it should have at least one function in it
        if 'def ' in response:
            extracted = parse_code_from_response(response)
            code_or_requirement = 'code'
        # If it's not python code, it's probably requirements
        else:
            extracted = parse_modified_user_requirements_from_response(response)
            code_or_requirement = 'requirements'
    # If there are |||, it's probably requirements
    elif '|||' in response:
        extracted = parse_modified_user_requirements_from_response(response)
        code_or_requirement = 'requirements'
    else:
        extracted = None
    return extracted, code_or_requirement
            



def parse_code_from_response(response):

    """
    Returns the code from the response from LLM.
    In the prompt to code, we have asked the LLM to return the code inside triple backticks.

    Args:
    - response (str): The response from LLM

    Returns:
    - matches (list): A list of strings with the code

    """

    pattern = r"```python([\s\S]*?)```"
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        matches = '\n'.join(matches)
    else:
        matches = matches[0]
    return matches

def parse_modified_user_requirements_from_response(response):
    
    """
    Returns the modified user requirements from the response from LLM. 
    In the prompt to modify, we have asked the LLM to return the modified user requirements inside triple pipe delimiters.

    Args:
    - response (str): The response from LLM

    Returns:
    - matches (list): A list of strings with the modified user requirements

    """
    if '|||' in response:
        pattern = r"\|\|\|([\s\S]*?)\|\|\|"
    # It shouldnt have ```python in it
    pattern = r"```([\s\S]*?)```"
    matches = re.findall(pattern, response)
    # if there are multiple matches, join by new line
    if len(matches) > 0:
        matches = '\n'.join(matches)
    else:
        matches = matches[0]
    return matches
