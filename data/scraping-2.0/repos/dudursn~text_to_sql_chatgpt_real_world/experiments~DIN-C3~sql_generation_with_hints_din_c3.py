import os
import sys
from langchain.callbacks import get_openai_callback
import time
path = os.path.abspath('')
module_path = os.path.join(path, "..\\C3")
if module_path not in sys.path:
    sys.path.append(module_path)

    
from c3_calibration_with_hints import generate_calibration_with_hints

path = os.path.abspath('')
module_path = os.path.join(path, "..\\DIN")
if module_path not in sys.path:
    sys.path.append(module_path)
    
from din_generating_sql_by_type import generating_sql_by_type_prompt_maker


def generating_sql_with_hints(db, llm, question, schema_links, classification, tables=[], callback=None):

    template = generating_sql_by_type_prompt_maker(db, classification, schema_links, tables=tables)
    template = template.format(question=question)
    messages = generate_calibration_with_hints(template)
    SQL = generate(llm, messages, callback)
    return SQL
    
def generate(llm, messages, callback=None):
    response = None
    while response is None:
        try:
            with get_openai_callback() as cb:
                response = llm.generate(messages)
                if response is not None:
                    if callback is not None:
                        callback({"sql_generation_din_c3":cb})
        except:
            time.sleep(3)
            pass
    try:
        SQL = response.generations[0][0].text
        SQL = SQL.split("SQL: ")[1]
    except:
        print("SQL slicing error")
        SQL = response.generations[0][0].text
    print('SQL Parcial: ', SQL)
    return SQL

