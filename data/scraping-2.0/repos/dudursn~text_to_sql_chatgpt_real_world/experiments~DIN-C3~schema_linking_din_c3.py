import os
import sys
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import time

path = os.path.abspath('')
module_path = os.path.join(path, "..\\C3")
if module_path not in sys.path:
    sys.path.append(module_path)
    
from table_recall_module import table_recall_main
from column_recall_module import column_recall_main
from prompts_template.schema_linking_template import SCHEMA_LINKING_PROMPT

TIME_TO_SLEEP = 4

def generate(llm, data_input, prompt, callback=None):
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = None
    while response is None:
        try:
            with get_openai_callback() as cb:
                response = llm_chain.generate(data_input)
                if response is not None:
                    if callback is not None:
                        callback({"schema_linking":cb})
        except:
            time.sleep(TIME_TO_SLEEP)
            pass
    try:
        schema_links = response.generations[0][0].text
        schema_links = schema_links.split("Schema_links: ")[1]
    except:
        print("Slicing error for the schema_linking module")
        schema_links = "[]"
        
    return schema_links

def schema_linking_din_c3(question, db, llm_c3, llm_din, add_fk = False, callback=None):
    
    table_schema = db.get_schema_openai_prompt()
    tables_ori = db.get_table_names()
    table_list = table_recall_main(table_schema, tables_ori, question, llm_c3, callback= callback)

    specific_tables_schema = db.get_schema_openai_prompt(table_list)
    tables_cols_ori = db.get_schema_json(table_list)
    foreign_keys_prompt = ""
    if add_fk:
        foreign_keys_prompt = db.get_foreign_keys_openai_prompt(table_list)

    schema_result = column_recall_main(specific_tables_schema, tables_cols_ori, question, llm_c3, foreign_keys_prompt, callback=callback)
    
    if schema_result is not None:
        schema_result_prompt = get_schema_to_clear_prompt(schema_result, foreign_keys_prompt)
    else:
        schema_result_prompt = db.get_schema_openai_prompt(table_list) + foreign_keys_prompt
            
    template = SCHEMA_LINKING_PROMPT.format(schema=schema_result_prompt, question="{question}")
    prompt = PromptTemplate(template=template, input_variables=["question"])
    data_input = [{"question": question}]
    
    schema_linking_result = generate(llm_din, data_input, prompt, callback=callback)
    return schema_linking_result, table_list

def get_schema_to_clear_prompt(schema_result, foreign_keys_prompt):
    schema_result_pompt = ""
    for tbl, columns in schema_result.items():
        schema_result_pompt += f"# {tbl} ({', '.join([c for c in columns])})\n"
    
    schema_result_pompt += foreign_keys_prompt
        
    return schema_result_pompt

