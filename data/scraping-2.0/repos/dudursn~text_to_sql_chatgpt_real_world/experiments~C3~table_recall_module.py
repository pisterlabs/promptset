from collections import Counter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import time

TIME_TO_SLEEP = 6

TABLE_RECALL_PROMPT = """
Given the database schema and question, perform the following actions: 
1 - Rank all the tables based on the possibility of being used in the SQL according to the question from the most relevant to the 
least relevant, Table or its column that matches more with the question words is highly relevant and must be placed ahead.
2 - Check whether you consider all the tables.
3 - Output a list object in the order of step 2, Your output should contain all the tables. The format should be like: 
[
    "table_1", "table_2", ...
]

Schema: 
{schema}

Question:
{question}
"""

def table_recall_main(schema, tables_ori, question, llm, callback=None):
    
    template = TABLE_RECALL_PROMPT.format(schema=schema, question='{question}')
    prompt = PromptTemplate(template=template, input_variables=["question"])
    data_input = [{"question": question}]
    tables_all = generate(llm, data_input, prompt, callback)
    table_list = table_self_consistency(tables_all, tables_ori)
    return table_list
    

def generate(llm, data_input, prompt, callback=None):
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    tables_all = None
    while tables_all is None:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.generate(data_input)
                tables_all = get_tables_response(result)
                if tables_all is not None:
                    if callback is not None:
                        callback({"table_recall":cb})
                else:
                    time.sleep(TIME_TO_SLEEP)
        except:
            print(f'api error, wait for {TIME_TO_SLEEP} seconds and retry...')
            time.sleep(TIME_TO_SLEEP)       
    
    return tables_all
        
        

def get_tables_response(responses):
    
    all_tables = []
    for table_response in responses.generations[0]:
        raw_table = table_response.text
        try:
            raw_table = '[' + raw_table.split('[', 1)[1]
            raw_table = raw_table.rsplit(']', 1)[0] + ']'
            raw_table = eval(raw_table)
            if Ellipsis in raw_table:
                raw_table.remove(Ellipsis)
        except:
            print('list error')
            return None
       
        all_tables.append(raw_table)

    return all_tables
    
def table_self_consistency(tables_all, tables_ori):
    
    tables_sc = []
    for id, tables in enumerate(tables_all):
        tables_exist = []
        for table in tables:
            if table.lower() in tables_ori:
                tables_exist.append(table.lower())
                if len(tables_exist) == 4:
                    break
            tables_sc.append(tables_exist)
    counts = Counter(tuple(sorted(lst)) for lst in tables_sc)

    most_list, count = counts.most_common(1)[0]

    for table_list in tables_sc:
        if sorted(table_list) == list(most_list):
            return table_list
 