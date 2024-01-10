from prompts.classification_prompt import *
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
                print('-- Classification -- ')
                print(response)
                if response is not None:
                    if callback is not None:
                        callback({"classification":cb})
        except:
            time.sleep(TIME_TO_SLEEP)
            pass
    try:
        classification_label = response.generations[0][0].text
        predicted_class = classification_label.split("Label: ")[1]
    except:
        print("Slicing error for the classification module")
        predicted_class = '"EASY"'
    
    classification = {'predicted_class': predicted_class, 'classification_label': classification_label}
        
    return classification

def classification_prompt_maker(db, schema_links, tables=[]):
    schema_basic_prompt = db.get_schema_basic_prompt(tables)
    fk_basic_prompt = db.get_foreign_keys_basic_prompt(tables)

    template = classification_prompt.format(schema=schema_basic_prompt, 
                                            foreign_keys=fk_basic_prompt,
                                            schema_links=schema_links,
                                            question="{question}")
    
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt


def classification_module(db, llm, question, schema_links, tables=[], callback=None):
    prompt = classification_prompt_maker(db, schema_links, tables=tables)
    data_input = [{"question": question}]
    classification_label = generate(llm, data_input, prompt, callback=callback)
    return classification_label
