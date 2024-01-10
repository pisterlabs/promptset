###############################################################################################################
#     Import Libraries                                                                                        #
###############################################################################################################

# OS related libraries
import os
from os import path
from time import sleep

# openAI API related functions (and login)
import openai
from openai.embeddings_utils import distances_from_embeddings

# running secrets
with open('./secrets.sh') as f:
    os.environ.update(
        line.replace('export ', '', 1).strip().split('=', 1) for line in f
        if 'export' in line
)

openai.api_key = os.getenv("OPENAI_API_KEY")

# basic ETL libraries
import pandas as pd
import numpy as np

# parameters
from .config import MODEL_PARAMS, PROMPT_BASE_DICT
from .preprocess_utils import _get_embedding, pprint


###############################################################################################################
# calling the OpenAI API

def _llm_call(prompt_dict: dict, model: str='gpt4', 
              version_num: int=1, maxretry_num: int=5) -> dict:
    """
    Calling a large language model, described in the model variable. The prompt dict
    should contain a prompt with the key of the selected model.
    
    """
    
    # generating base parameter dictionary
    _prompt = prompt_dict[model]['prompt']

    if model in ['davinci', 'curie']:
        _model_params = {'prompt': _prompt} # type: ignore - covered in all cases
    elif model in ['gpt3.5', 'gpt4']:
        _model_params = {'messages': _prompt} # type: ignore - covered in all cases
    else:
        raise ValueError(f'Unknown model: {model}')
    
    # adding extra parameters
    _model_params.update(MODEL_PARAMS[model])
    if version_num > 1:
        _model_params.update({'n': version_num})
    
    # CALLING THE LLM
    _llm_answer = {}

    # openai ChatCompletion cases
    if model in ['gpt3.5', 'gpt4']:
        
        # creating an empty answer, init mid-variables
        _llm_answer = {'choices':[{'message':{'content':''}}]}
        _curr_try = 0
        _solved = False

        # trying to get an answer
        while ((not _solved) and (_curr_try < maxretry_num)):
            try:
                _llm_answer = openai.ChatCompletion.create(**_model_params)
                _solved = True
            except Exception as e:
                print(f"The following error catched while calling Open AI API: {e}. Retrying in 1 second...")
                sleep(1)
                _curr_try += 1
    
    elif model in ['davinci', 'curie']:

        # creating an empty answer, init mid-variables
        _llm_answer = {'choices':[{'text':''}]}
        _curr_try = 0
        _solved = False

        # trying to get an answer
        while ((not _solved) and (_curr_try < maxretry_num)):
            try:
                _llm_answer = openai.Completion.create(**_model_params)
                _solved = True
            except Exception as e:
                print(f"The following error catched while calling Open AI API: {e}. Retrying in 1 second...")
                sleep(1)
                _curr_try += 1
    
    else:
        raise ValueError(f'Unknown model: {model}')
    
    # returning the answer
    return _llm_answer


###############################################################################################################
# context generation

def _vectorize_question(in_question: str, maxretry_num: int=5) -> np.array:
    """
    This function vectorizes the question, and returns it.
    """

    _vectorized_question = np.array([])
    _curr_try = 0
    _solved = False

    # trying to get an answer
    while ((not _solved) and (_curr_try < maxretry_num)):
        try:
            _vectorized_question = _get_embedding(in_text=in_question, model='openai', _sleeptime=0)
            _solved = True
        except Exception as e:
            print(f"The following error catched while calling Open AI API: {e}. Retrying in 1 second...")
            sleep(1)
            _curr_try += 1

    return _vectorized_question


def _find_context(in_df_knowledge_base: pd.DataFrame, vectorized_question: np.array, max_len: int=1500,
                  _verbose: bool=False) -> str:
    """
    This function finds the context of the question, and returns it.
    """

    _context = ""
    in_df_embeddings = in_df_knowledge_base.copy()
    
    # calculating distance from the question, ordering the table
    in_df_embeddings['q_distance'] = distances_from_embeddings(
        vectorized_question, in_df_embeddings['node_embedding'].values, distance_metric='cosine')
    in_df_embeddings = in_df_embeddings.sort_values(by='q_distance', ascending=True)

    # getting the context (until the limit)
    _curr_len = 0
    _saved_length = 0
    for _idx, _row in in_df_embeddings.iterrows():
        _saved_length = _curr_len
        _curr_len += _row['n_tokens'] + 4
        if (_curr_len < max_len) or _curr_len < 5:
            _context += _row['split_text'] + "\n\n##############################\n\n"
        else:
            break
    
    if _verbose:
        pprint("{} long context is generated.".format(_saved_length))

    return _context


def _prompt_generation(in_question: str, in_context: str, project_name: str, model: str='gpt-4') -> dict:
    """
    This function generates a prompt dictionary, and returns with it.
    """

    # giving a basic return dictionary
    _return_dict = {
        'davinci' : {'prompt':''}, 
        'gpt3.5' : {'prompt':[]}, 
        'gpt4' : {'prompt':[]}, 
        'curie' : {'prompt':''}
    }
    
    # prompt for non-chatting models
    if model in ['davinci', 'curie']:

        _prompt_text = ""
        
        # adding the system messages
        for _sysmsg in PROMPT_BASE_DICT[project_name]['system_messages']:
            _prompt_text += _sysmsg + "\n\n"
        
        # adding the context and the question
        _prompt_text = _prompt_text + "\n\nContext:\n\n{}\n\nThe question to answer from the context above:{}\n\nMy polite answer: ".format(
            in_context, in_question)
        
        _return_dict[model].update({'prompt': _prompt_text})
    
    # prompt for chatting models
    elif model in ['gpt3.5', 'gpt4']:

        _messages = []

        # adding the system messages
        for _sysmsg in PROMPT_BASE_DICT[project_name]['system_messages']:
            _messages.append({'role': 'system', 'content': _sysmsg})
        
        # adding the context and the question
        _messages.append({'role': 'system', 'content': "Answer the question from the following context:\n\n{}".format(in_context)})
        _messages.append({'role': 'user', 'content': in_question})

        _return_dict[model].update({'prompt': _messages})
    
    return _return_dict


###############################################################################################################
#     Main functions                                                                                          #
###############################################################################################################

def answer_question(path_in_knowledge_base: str, in_question: str, project_name: str, model: str='gpt4', 
                    max_context_len: int=1500, _verbose=False) -> str:
    """
    This function answers a question.
    """

    # reading the knowledge base
    _df_knowledge_base = pd.read_pickle(path_in_knowledge_base)

    if _verbose:
        pprint("Knowledge base loaded with {} lines.".format(_df_knowledge_base.shape[0]))

    # vectorizing the question
    _vectorized_question = _vectorize_question(in_question=in_question)

    if _verbose:
        pprint("Question is vectorized. Dimension: {}.".format(len(_vectorized_question)))

    # finding the context
    _context = "" 
    if len(_vectorized_question) > 0:
        _context = _find_context(in_df_knowledge_base=_df_knowledge_base, vectorized_question=_vectorized_question, 
                                 max_len=max_context_len, _verbose=_verbose)
    
    # generating the prompt
    _prompt_dict = {}
    if len(_context) > 0:
        _prompt_dict = _prompt_generation(in_question=in_question, in_context=_context, project_name=project_name, 
                                          model=model)
    
    # calling the model
    _llm_answer = ""
    if len(_context) > 0:
        _llm_answer = _llm_call(prompt_dict=_prompt_dict, model=model, version_num=1, maxretry_num=5)
    
        if model in ['gpt3.5', 'gpt4']:
            _llm_answer = _llm_answer['choices'][0]['message']['content']
        else:
            _llm_answer = _llm_answer['choices'][0]['text']

    # returning the answer
    if len(_llm_answer) > 0:
        _ret_answ = _llm_answer
    else:
        _ret_answ = "Error: please try again: your question was empty or the model had errors."
    
    return _ret_answ

