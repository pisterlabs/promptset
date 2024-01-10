
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from pprint import pprint
import json
import pandas as pd
import glob
import re

# extract setence text string list from cv-corpus dataset
def load_setences(file_path:str = 'cv-corpus/ko/validated.tsv', 
                  scolumn_name:str = 'sentence',
                  vcolumn_name:str = 'path') -> Tuple[List[str], List[str]]:
    
    # load full data into dataframe
    if(file_path.endswith('.tsv')):
        df = pd.read_csv(file_path, sep='\t')
    
    elif (file_path.endswith('.csv')):
        df = pd.read_csv(file_path)
    
    elif (file_path.endswith('.xlsx')):
        df = pd.read_excel(file_path)

    # extract sentences from dataframe
    sentences = df[scolumn_name].tolist()
    
    # extract voice file path from dataframe
    voice_paths = df[vcolumn_name].tolist()

    # return sentence list
    return sentences, voice_paths


# make user prompt template from sentence list 
def make_user_prompt(src_sentences:List[str]) -> Tuple[str, Dict[str, str]]:

    # make user prompt object and string 
    user_prompt = {}
    user_prompt["src"] = src_sentences
    
    # escape curly brackets
    user_prompt_str = json.dumps(user_prompt).replace('{','{{').replace('}','}}')
    
    # return string type and object type user prompt
    return user_prompt_str, user_prompt


# make full prompt with user and system prompt 
def make_prompt_template(user_prompt_str:str, 
                         sys_prompt_file:str = 'prompts/1. translate.txt',) -> ChatPromptTemplate:
    
    # load system prompt
    with open(sys_prompt_file, 'r') as f:
        sys_prompt_str = f.read()

    # compose prompt template with system and user prompt
    prompt_template = ChatPromptTemplate.from_messages( [
        ("system", sys_prompt_str),
        ("human", user_prompt_str),
        ] )
    
    # return ChatPromptTemplate object
    return prompt_template


# GPT-4 설정
def make_llm_chat(temperature:float = 0.9, 
                  max_tokens:int = 2048, 
                  model:str = 'gpt-4', 
                  verbose:bool = True) -> ChatOpenAI:
    
    llm = ChatOpenAI(temperature=temperature, 
                     max_tokens=max_tokens, 
                     model=model,
                     verbose = verbose)
    
    # return ChatOpenAI object
    return llm


if __name__ == "__main__":

    # environment variables, esp. OPENAI_API_KEY
    load_dotenv()
    
    # paths for required files
    suite_file_path = 'cv-corpus/ko/validated.tsv'
    prmpt_file_path = 'prompts/1. translate.txt'

    # gpt configuration
    temperature = 0.9
    max_tokens = 2048
    model = 'gpt-4'

    # load sentences from cv-corpus dataset
    sntnc_list, vfile_list = load_setences(suite_file_path, 'sentence')

    # vars for batch request
    n_sntnc = len(sntnc_list)
    n_refs = 3     # how many translated setences required for a given setence? for now 3, is fixed
    n_batch = 50   # how many source setences to be requested to gtp at a time? 
    print(f'# Number of Setences to translate : {n_sntnc}')

    # translation direction
    src_lang = 'korean'
    dst_lang = 'english'

    # dataframe to save results
    columns = ['path', 'src_lang', 'dst_lang', 'sentence', 'translated']
    df = None
    # batch request

    # check if there is any result file from previsou request
    files = glob.glob('* - result.xlsx')

    if files == []:
        idx = 0
        cnt = 0
        df = pd.DataFrame(columns=columns)
    else:  # set begining index and count from previous result
        max_file = max(files, key=lambda file: int(re.search(r'(\d+) - result.xlsx', file).group(1)))
        max_int = int(re.search(r'(\d+) - result.xlsx', max_file).group(1))
        idx = max_int * n_batch # next index at restart
        cnt = max_int + 1 # next count(interim result file number) at restart     
        df = pd.read_excel(max_file)

    stop = False
    while (stop):
        eidx = idx + n_batch

        if( eidx > n_sntnc ): 
            eidx = idx + n_sntnc % n_batch 
            stop = True
        
        # build prompts
        s_batch = sntnc_list[idx:eidx]
        v_batch = vfile_list[idx:eidx]

        usr_prmpt_str, usr_prmpt = make_user_prompt(s_batch)
        sys_prmpt_tmpl = make_prompt_template(usr_prmpt_str, prmpt_file_path)
        
        # configure llm model
        llm = make_llm_chat(temperature, max_tokens, model)

        # prompt chaining
        chain = LLMChain(
            llm = llm, 
            prompt = sys_prmpt_tmpl, 
            output_key = "output",
            verbose = True
        )

        # request with additional info
        req = {}
        req['src_lang'] = src_lang
        req['dst_lang'] = dst_lang
        result_str = chain(req)

        # result Check
        result = json.loads(result_str['output'])
        pprint(result)

        # compose result into dataframe with other info
        src_lang = req['src_lang']  
        dst_lang = req['dst_lang']
        src_sentences = usr_prmpt['src']
        dst_sentences = result['dst']
        for i in range(len(src_sentences)):
            for j in range(3):
                new_row = pd.DataFrame({'src_lang': [src_lang],
                                        'dst_lang': [dst_lang],
                                        'path': [v_batch[i]],
                                        'sentence': [src_sentences[i]],
                                        'translated': [dst_sentences[i][j]]
                                        })
                df = pd.concat([df, new_row], ignore_index=True)
        print(df)

        # update index
        idx = idx + n_batch
        cnt = cnt + 1
        
        # save interim result into file
        df.to_excel(f'{cnt} - result.xlsx', index=False)

    # save filnal result into file
    df.to_excel('result.xlsx', index=False)