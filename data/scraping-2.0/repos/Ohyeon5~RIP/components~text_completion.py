import openai
import streamlit as st
from typing import Dict, Any, Tuple


""" for all possible parameters, see https://beta.openai.com/docs/api-reference/completions/create
Note that `best_of` >= `n` is required 
"""
_default_completion_parameters = {
    "temperature":0.7,
    "max_tokens":60,
    "top_p":1.0,
    "frequency_penalty":0.0,
    "presence_penalty":0.0,
    "n": 1,
    "stop":"\n",
    }

def get_default_completion_params() -> Dict[str, Any]:
    return _default_completion_parameters

def set_completion_params(param_dict: Dict[str, Any]) -> Dict[str, Any]:
    params = get_default_completion_params()
    return {k:  param_dict[k] if k in param_dict.keys() else v for k,v in params.items()}

@st.cache(suppress_st_warning=True, persist=True)
def transform_text_to_easy_text(text_input: str, **kwags) -> str:
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Summarize this for a second-grade student:\n\n{text_input}",
        **kwags,
    )
    return response.choices[0].text

@st.cache(suppress_st_warning=True, persist=True)
def transform_text_to_scene_description(text_input: str, **kwags) -> str:
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Here is a concept:\n\n{text_input} \n\nDescribe an image that represents this scene:",
        **kwags,
    )
    return response.choices[0].text

def get_semantic_search_results(question: str)-> Tuple[str, str]:
    """
    get semantic search results from openAI api and return both search result and keywords

    args:
        question (str): question to ask
    returns:
        result (str): search result
        keywords (str): summarized results
    """    
    whoami = "I am a highly intelligent question answering bot against lowering CO2 emission. If you comment about your plan for popular eco-friendly solutions, I will answer with the side effects of those."
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=whoami + "\n\nQuestion:" + question + "\n\nAnswer:",
      temperature=0,
      max_tokens=4000,
      top_p=1,
      frequency_penalty=0.1,
      presence_penalty=0,
    )
    response_summary = openai.Completion.create(
      model="text-davinci-002",
      prompt="Summarize this with keywords:\n\n" + response.choices[0].text,
      temperature=0,
      max_tokens=50,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
    )
    return response.choices[0].text, response_summary.choices[0].text

def get_suggested_actions(question: str, summaries: str):

    whoami3 = "I'm a climate change campaigner to help to make actions that the public can follow easily. I also consider which scenario you are in. If you tell me the potential risks and the scenario, I will let you know which actions to take."
    risks = summaries
    scenario = question
    response_actions = openai.Completion.create(
      model="text-davinci-002",
      prompt=whoami3+" \n\nPotential risks:\n"+risks+"\n\nScenario:\n"+scenario+"\n\nAction:\n",
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
    )
    return response_actions.choices[0].text