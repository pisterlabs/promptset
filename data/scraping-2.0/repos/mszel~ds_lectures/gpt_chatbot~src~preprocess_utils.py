###############################################################################################################
#     Import Libraries                                                                                        #
###############################################################################################################

# OS related libraries
import os
import shutil
import time
from os import path
from time import sleep
import warnings
warnings.simplefilter("ignore", UserWarning)

# NLP-related libraries
import tiktoken
TOKENIZER = tiktoken.get_encoding("cl100k_base")

import re

# pdf cleaning
from PyPDF2 import PdfReader

# openAI API related functions (and login)
import openai

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
from tqdm import tqdm
tqdm.pandas()

###############################################################################################################
# util functions

def pprint(s: str) -> None:
    """ 
    Pretty print: writing out the time while print the input string.
    """
    print('[' + time.strftime('%a %H:%M:%S') + '] ' + s)
    pass

###############################################################################################################
# document preprocessing functions

def _string_cleaner(in_str: str, spec_chars: list=[]) -> str:
    """
    Cleaning the string from non-unicode and non alphanumeric characters.
    Keeping the listed characters from the spec_chars input list.
    """

    _specdef = ['@', '#', '$', '%', '&', '*', '(', ')', '/', '"', '\'', 
                '「', '」', '|', '-', ':', ' ', ',', '.', '!', '?', '[', 
                ']', '{', '}', '<', '>', '=', '+', '~', '`', '^', ';', 
                '\n', '。', '、', ',', '\t'] + spec_chars

    _goodchars = re.findall("[^\W]", in_str, re.UNICODE)
    _goodchars = list(set(_specdef + _goodchars))

    return ''.join([_i for _i in in_str if _i in _goodchars])


def _rawtext_cleaner(in_str: str, spec_chars: list=[]) -> str:
    """
    Removing extra new lines, tabs, white spaces, etc - as well as the
    non-unicode characters.
    """
    
    _return_txt = ""
    _raw_text = re.sub(r'\n{2,}', '\n\n', in_str)
    _raw_text = re.sub(r'[ \t]+', ' ', _raw_text)
    _raw_text = _raw_text.split('\n\n')
    _raw_text = [_string_cleaner(_i, spec_chars) for _i in _raw_text]
    _raw_text = [_i for _i in _raw_text if len(_i.strip()) >= 2]
    
    if len(_raw_text) > 0:
        _return_txt = '\n\n'.join(_raw_text)
    
    return _return_txt


def _handle_text(path_in_txt: str, _encoding: str, spec_chars: list=[]) -> str:
    """
    Handling text files.
    """

    with open(path_in_txt, encoding=_encoding, errors='ignore') as f:
        _raw_text = '\n'.join(f.readlines())
    
    # do the basic cleaning
    _raw_text = _rawtext_cleaner(_raw_text, spec_chars)
    return _raw_text


def _handle_pdf(path_in_pdf: str, spec_chars: list=[]) -> str:
    """
    Handle pdf files.
    """
    _raw_text = ''

    pdffileobj=open(path_in_pdf, 'rb')
    _allpages = PdfReader(pdffileobj)
    _totpages = _allpages.numPages
    for _page in range(_totpages):
        pagehandle = _allpages.getPage(_page)
        _raw_text += pagehandle.extractText() + '\n\n'
    
    # do the basic cleaning
    _raw_text = _rawtext_cleaner(_raw_text, spec_chars)
    return _raw_text


###############################################################################################################
# embedding functions

def _get_embedding(in_text: str, model: str='openai', _sleeptime: float=0.1, maxretry_num: int=5) -> np.array:
    """
    Get the embedding of a text using openAI's models.
    """

    _return_val = np.array([])
    if len(in_text.replace("\n", " ").strip()) > 0:
        if model == 'openai':
            text_m = in_text.replace("\n", " ")
            _curr_try = 0
            _solved = False

            # trying to get an answer
            while ((not _solved) and (_curr_try < maxretry_num)):
                try:
                    sleep(_sleeptime) # to avoid openAI error (429-alike)
                    _return_val = openai.Embedding.create(input=text_m, model='text-embedding-ada-002')['data'][0]['embedding']
                    _solved = True
                except Exception as e:
                    pprint('Error while calling openAI API... Trying in 1 sec...')
                    print(e)
                    _curr_try += 1
        else:
            raise ValueError('Model not supported.')

    return _return_val


###############################################################################################################
# text splitting functions

def _split_text_basic(path_in_txt: str, _encoding: str='latin1', min_token_size: int=128, 
                      max_token_size:int=512) -> pd.DataFrame:
    """
    Read a text by paragraphs into a data frame and do the following steps:
     - split by paragraphs
     - if a text is too small, append it to the next paragraph
     - after checking the newly created paragraphs, if a paragraph is too big, split it into
       n smaller paragraphs (by dots if possible).
    """

    tokenizer = TOKENIZER
    _ret_df = pd.DataFrame(columns=['split_text', 'n_tokens'])

    with open(path_in_txt, encoding=_encoding, errors='ignore') as f:
        _raw_text = '\n'.join(f.readlines())
    
    # splitting by paragraphs and make sure there is no smaller paragraph created
    _text_col = _raw_text.split('\n\n')
    _text_col_cl = []

    _curr_text = ''
    _curr_len = 0
    for _i in range(len(_text_col)):
        _curr_len = _curr_len + len(tokenizer.encode(_text_col[_i]))
        _curr_text = _curr_text + _text_col[_i] + '\n\n'
        if _curr_len > min_token_size:
            _text_col_cl.append(_curr_text)
            _curr_text = ''
            _curr_len = 0
    
    # adding the leftover
    if len(_curr_text) > 0:
        _text_col_cl.append(_curr_text)
    
    # splitting by sentences (within the paragraphs - list of lists)
    for _i in range(len(_text_col_cl)):
        _text_col_cl[_i] = _text_col_cl[_i].replace('\n', '\n###sep###').replace('. ', '.###sep###').split('###sep###')
    
    # Where the paragraph is small enough keep. Otherwise split by sentences.
    _text_col_final = []

    for _pg in _text_col_cl:
        if len(tokenizer.encode(' '.join(_pg))) < max_token_size:
            _text_col_final.append(' '.join(_pg))
        else:
            _curr_text = ''
            _curr_len = 0

            for _i in range(len(_pg)):
                if _curr_len + len(tokenizer.encode(_pg[_i])) > max_token_size:
                    _text_col_final.append(_curr_text)
                    _curr_text = _pg[_i]
                    _curr_len = len(tokenizer.encode(_pg[_i]))
                else:
                    _curr_text = _curr_text + ' ' + _pg[_i]
                    _curr_len = _curr_len + len(tokenizer.encode(_pg[_i]))
            
            if len(_curr_text) > 0:
                _text_col_final.append(_curr_text)
    
    # removing the empty paragraphs
    _text_col_final = [x for x in _text_col_final if len(x) > 2]

    _out_df = pd.DataFrame({'split_text': _text_col_final})
    _out_df['n_tokens'] = _out_df.split_text.apply(lambda x: len(tokenizer.encode(x)))

    if len(_out_df) > 0:
        _ret_df = _out_df.copy()
    
    return _ret_df




###############################################################################################################
#     Main functions                                                                                          #
###############################################################################################################


def page_preprocessor(path_in_folder: str, path_out_folder: str, _encoding: str='latin1', _verbose: bool=True) -> None:
    """
    Converts the txt and pdf files from a folder to simple, clean text files.
    """
    
    # create or clear the output folder
    if path.exists(path_out_folder):
        shutil.rmtree(path_out_folder)
    os.makedirs(path_out_folder, exist_ok=True)

    # clean all txt and pdf files
    _files = [_i for _i in os.listdir(path_in_folder) if _i.endswith('.txt') or _i.endswith('.pdf')]

    for _file in tqdm(_files, desc='Cleaning and converting files...'):
        _source = path.join(path_in_folder, _file)
        if _verbose:
            pprint('Processing file: ' + _source)

        if _file.endswith('.txt'):
            _target = ''.join(_file.split('.txt')[:-1]) + '_cl.txt'
            _target = path.join(path_out_folder, _target)

            _raw_text = _handle_text(_source, _encoding=_encoding)
            with open(_target, 'w', encoding=_encoding) as f:
                f.write(_raw_text)
        
        if _file.endswith('.pdf'):
            _target = ''.join(_file.split('.pdf')[:-1]) + '_conv_pdf.txt'
            _target = path.join(path_out_folder, _target)

            _raw_text = _handle_pdf(_source)
            with open(_target, 'w', encoding=_encoding) as f:
                f.write(_raw_text)
    
    pass
    

def knowledge_base_maker(path_in_folder: str, path_out_folder: str, _encoding: str='latin1', min_token_size: int=128, 
                         max_token_size:int=512, model: str='openai', _sleeptime: float=0.1, _verbose: bool=True) -> None:
    """
    Creates a knowledge base from the cleaned text files.
    """

    # if the old file exists, read the old file, if the out folder does not exist, create it
    if path.exists(path.join(path_out_folder, 'knowledge_base.pickle')):
        _old_df = pd.read_pickle(path.join(path_out_folder, 'knowledge_base.pickle'))
    else:
        _old_df = pd.DataFrame(columns=['split_text', 'n_tokens', 'file_source', 'node_embedding'])
        os.makedirs(path_out_folder, exist_ok=True)
    
    # convert the files to data frames (by splittext)
    _files = [_i for _i in os.listdir(path_in_folder) if _i.endswith('_cl.txt') or _i.endswith('_conv_pdf.txt')]
    _df_list = []

    for _file in tqdm(_files, desc='Converting files to knowledge base...'):
        _tmp_df = _split_text_basic(
            path_in_txt=path.join(path_in_folder, _file), 
            _encoding=_encoding, min_token_size=min_token_size, 
            max_token_size=max_token_size)
        
        _tmp_df['file_source'] = _file
        _df_list.append(_tmp_df[['split_text', 'n_tokens', 'file_source']])
    
    if len(_df_list) > 0:

        # create the new data frame (base of the knowledge base)
        _new_df = pd.concat(_df_list, axis=0).reset_index(drop=True)

        # if possible, adding the embeddings from the old data frame
        _new_df_o = _new_df[_new_df.split_text.isin(_old_df.split_text.unique())].copy()
        _new_df_n = _new_df[~(_new_df.split_text.isin(_old_df.split_text.unique()))].copy()

        _helper_old = _old_df[['split_text', 'node_embedding']].drop_duplicates(subset=['split_text'])
        _new_df_o = pd.merge(_new_df_o, _helper_old, how='inner', on=['split_text'])

        # if not, calling open AI API
        _new_df_n['node_embedding'] = _new_df_n.split_text.progress_apply(lambda x: _get_embedding(x, model=model, _sleeptime=_sleeptime))

        # concatenating the new and old data frames
        _new_df = pd.concat([_new_df_o, _new_df_n], axis=0).reset_index(drop=True)

        # saving the data frame
        _new_df.to_pickle(path.join(path_out_folder, 'knowledge_base.pickle'))

        if _verbose:
            pprint('Knowledge base created with ' + str(len(_new_df)) + ' nodes.')
    
    pass