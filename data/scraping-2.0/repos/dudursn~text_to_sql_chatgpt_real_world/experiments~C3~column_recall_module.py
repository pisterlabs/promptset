from collections import Counter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import time
import json

 
TIME_TO_SLEEP = 6

COLUMN_RECALL_PROMPT = """
Given the database tables and question, perform the following actions: 
1 - Rank the columns in each table based on the possibility of being used in the SQL, Column that matches more with the question 
words or the foreign key is highly relevant and must be placed ahead. You should output them in the order of the most 
relevant to the least relevant.
Explain why you choose each column.
2 - Output a JSON object that contains all the columns in each table according to your explanation. The format should be like: 
{{
    "table_1": ["column_1", "column_2", ......], 
    "table_2": ["column_1", "column_2", ......],
    "table_3": ["column_1", "column_2", ......],
    ......
}}

Schema: 
{schema}
{foreign_keys}

Question:
###  {question}
"""

def column_recall_main(schema, tabs_cols_ori, question, llm, foreign_keys_prompt, callback=None):
    
    prompt = PromptTemplate(template=COLUMN_RECALL_PROMPT, input_variables=["schema", "foreign_keys", "question"])
    fk_prompt = "Foreign keys:\n" + foreign_keys_prompt if len(foreign_keys_prompt)>0 else foreign_keys_prompt
    data_input = [{"schema":schema, "foreign_keys":fk_prompt, "question": question}]
    tabs_cols_all = generate(llm, data_input, prompt, callback=callback)
    if tabs_cols_all is None:
        return None
    return column_self_consistency(tabs_cols_all, tabs_cols_ori)

    

def generate(llm, data_input, prompt, callback=None):
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    tabs_cols_all = None
    attempts = 1
    
    while tabs_cols_all is None and attempts <= 3:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.generate(data_input)
                tabs_cols_all = get_tables_column_response(result)
                if tabs_cols_all is not None:
                    if callback is not None:
                        callback({"column_recall":cb})
                else:
                    time.sleep(TIME_TO_SLEEP)
            
        except:
            print(f'api error, wait for {TIME_TO_SLEEP} seconds and retry...')
            time.sleep(TIME_TO_SLEEP)
            pass
        finally:
            print(f"Column recall attempt: {attempts}")
            attempts += 1
    
    return tabs_cols_all
        
        

def get_tables_column_response(responses):
    
    tabs_cols_all = []
    for tabs_cols_response in responses.generations[0]:
        raw_tab_col = tabs_cols_response.text
        try:
            raw_tab_col = '{' + raw_tab_col.split('{', 1)[1]
            raw_tab_col = raw_tab_col.rsplit('}', 1)[0] + '}'
            raw_tab_col = json.loads(raw_tab_col)
        except:
            print('list error column recall')
            return None
        tabs_cols_all.append(raw_tab_col)
    return tabs_cols_all


def column_self_consistency(tabs_cols_all, tabs_cols_ori):

    candidates = {}
    results = {}
    for key in tabs_cols_ori:
        candidates[key] = []

    # filter out invalid tables
    for tabs_cols in tabs_cols_all:
        for key, value in tabs_cols.items():
            if key in tabs_cols_ori:
                candidates[key].append(value)

    for tab, cols_all in candidates.items():
        cols_ori = [item.lower() for item in tabs_cols_ori[tab]]
        cols_sc = []
        for cols in cols_all:
            cols_exist = []
            for col in cols:
                if col.lower() in cols_ori:
                    cols_exist.append(col)
                    if len(cols_exist) == 4:
                        break
            if len(cols_exist) > 0:
                cols_sc.append(cols_exist)
        # choose the top-5 columns with the highest frequency
        if len(cols_sc) > 0:
            cols_add = []
            for cols in cols_sc:
                cols_add = cols_add + cols
            counter = Counter(cols_add)
            most_common_cols = counter.most_common(5)
            temp = []
            for value, count in most_common_cols:
                temp.append(value)
            results[tab] = temp
        else:
            results[tab] = []
    
    return results