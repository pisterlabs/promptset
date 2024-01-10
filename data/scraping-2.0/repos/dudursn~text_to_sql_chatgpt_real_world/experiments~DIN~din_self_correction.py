from prompts.self_correction_prompt import *
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
                        callback({"self_correction":cb})
        except:
            time.sleep(TIME_TO_SLEEP)
            pass
   
    debugged_SQL = response.generations[0][0].text.replace("\n", " ")
    SQL = "SELECT " + debugged_SQL
    return SQL

def self_correction_prompt_maker(db):
    schema_basic_prompt = db.get_schema_basic_prompt()
    fk_basic_prompt = db.get_foreign_keys_basic_prompt()
    pk_basic_prompt = db.get_primary_keys_basic_prompt()
    template = self_correction_prompt.format(schema=schema_basic_prompt, 
                                             primary_keys=pk_basic_prompt,
                                             foreign_keys=fk_basic_prompt,
                                             dialect=db.get_dialect(),
                                             question='{question}',
                                             query='{query}')
    prompt = PromptTemplate(template=template, input_variables=['question', 'query'])
    return prompt


def self_correction_module(db, llm, question, sql, callback=None):
    prompt = self_correction_prompt_maker(db)
    data_input = [{'question': question, 'query': sql}]
    sql_query_fix = generate(llm, data_input, prompt, callback=callback)
    return sql_query_fix

