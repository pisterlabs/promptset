from prompts.generating_sql_by_type_prompt import *
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from din_parameter import TIME_TO_SLEEP
import time

def generate(llm, data_input, prompt, callback=None):
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = None
    while response is None:
        try:
            with get_openai_callback() as cb:
                response = llm_chain.generate(data_input)
                if response is not None:
                    if callback is not None:
                        callback({"sql_generation":cb})
        except:
            time.sleep(TIME_TO_SLEEP)
            pass
    try:
        SQL = response.generations[0][0].text
        SQL = SQL.split("SQL: ")[1]
    except:
        print("SQL slicing error")
        SQL = "SELECT"
    print(SQL)
    return SQL

def generating_sql_by_type_prompt_maker(db, classification, schema_links, tables =[]):
    
    predicted_class = classification['predicted_class']
    classification_label = classification['classification_label']
    schema_basic_prompt = db.get_schema_basic_prompt(tables)
    fk_basic_prompt = db.get_foreign_keys_basic_prompt(tables)
    
    if '"EASY"' in predicted_class:
        template = easy_prompt.format(schema=schema_basic_prompt, 
                                            schema_links=schema_links,
                                            question="{question}")
        print("EASY")
    elif '"NON-NESTED"' in predicted_class:
        template = medium_prompt.format(schema=schema_basic_prompt, 
                                            foreign_keys=fk_basic_prompt,
                                            schema_links=schema_links,
                                            question="{question}")
        print("NON-NESTED")
    else:
        sub_questions = classification_label.split('questions = ["')[1].split('"]')[0]
        
        template = hard_prompt.format(schema=schema_basic_prompt, 
                                            foreign_keys=fk_basic_prompt,
                                            schema_links=schema_links,
                                            question="{question}",
                                            sub_questions=sub_questions)
        print("NESTED")
        
    return template


def generating_sql_by_type_module(db, llm, question, schema_links, classification, tables=[], callback=None):
    template = generating_sql_by_type_prompt_maker(db, classification, schema_links, tables=tables)
    prompt = PromptTemplate(template=template, input_variables=["question"])
    data_input = [{"question": question}]
    SQL = generate(llm, data_input, prompt, callback=callback)
    return SQL

    