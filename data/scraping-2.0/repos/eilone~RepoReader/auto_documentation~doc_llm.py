import openai
from .llm_examples import EXAMPLES
from general_utils import (get_openai_api_key,)
from general_config import (
    LLM_TEMPERATURE,
    LLM_MODEL_NAME,
)

PROMPT_SYSTEM = """
You are a DBA and a documentation-generating machine, that gets sql logic of a table and its dependencies, 
and outputs a documentation for the table in YML format
"""

PROMPT_USER = """  
I have this undocumented sql model: Table name: {model[name]} 
Provide a comprehensive description of the table, based on the sql code in GBQ dialect, using the documentation of
its dependencies.

INSTRUCTIONS: 
1. The output should have a short general description about the purpose of the data, the granularity, the data source 
and whatever else you see fit. It is important to be descriptive and informative. 
2. THE OUTPUT FORMAT: a yml file. Remember that the output should be in the same format as the
dependencies' YML documentation, as provided in the context. 
3. Return descriptions for ALL the columns in the table. If you are not sure about the description, live an empty str.
4. If the SQL takes the column from the source without any changes, you can copy the description from the source's YML.
Else, you should describe the SQL logic of the column in the table.
For example, for a column that is being built using "SUM(src_metric) as final_metric" you should describe:
"the sum of src_metric from table X"
5. Important: keep the indentation and syntax of YML, same format as the dependencies' YML documentation, as provided.
{example_prompt[examples_instructions]}
            
              
            
            Table SQL: {model[code]}
            Dependencies YMLs: {deps}
            {example_prompt[examples]}
    """

def get_example_prompt(is_using_examples=True):
    if is_using_examples:
        return {
            "examples_instructions": "5. Learn from the examples provided below on how should a input-output look like",
            "examples": f'Examples: {EXAMPLES},'
        }
    else:
        return {
            "examples_instructions": "",
            "examples": "",
        }


def get_generated_doc(model, deps, is_using_examples=True, **kwargs):
    openai.api_key = get_openai_api_key()
    response = openai.ChatCompletion.create(
        model=kwargs.get('model_name', LLM_MODEL_NAME),
        temperature=kwargs.get('temperature', LLM_TEMPERATURE),
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": PROMPT_USER.format(
                model=model,
                deps=deps,
                examples=EXAMPLES,
                example_prompt=get_example_prompt(is_using_examples)
               )},
        ]
    )
    yml_doc = response['choices'][0]['message']['content'].strip()
    tokens = response['usage']['total_tokens']
    finish_reason = response['choices'][0]['finish_reason']

    return yml_doc, tokens, response, finish_reason
