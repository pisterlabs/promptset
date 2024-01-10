from prompts.schema_linking_prompt import *
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

def schema_linking_prompt_maker(db):
    schema_basic_prompt = db.get_schema_basic_prompt()
    fk_basic_prompt = db.get_foreign_keys_basic_prompt()
    template = schema_linking_prompt.format(schema=schema_basic_prompt, 
                                            foreign_keys=fk_basic_prompt,
                                            question="{question}")
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt


def schema_linking_module(db, llm, question, callback=None):
    prompt = schema_linking_prompt_maker(db)
    data_input = [{"question": question}]
    schema_links = generate(llm, data_input, prompt, callback=callback)
    return schema_links

