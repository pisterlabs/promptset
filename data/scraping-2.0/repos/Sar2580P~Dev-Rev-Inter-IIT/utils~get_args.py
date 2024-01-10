import sys , os
sys.path.append(os.getcwd())
from typing import List, Union, Any, Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.llm_utility import llm, small_llm
from memory.tool_memory import retrieve_tool_experience
from utils.templates_prompts import TOOLS_PROMPT_EXAMPLES, ARG_FILTER_PROMPT
import re, ast
from utils.tool_output_parser import parser
from icecream import ic
from utils.parsers import arg_filter_parser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import RetryWithErrorOutputParser


#-------------------------------------------------------------------------------------------------------------------------------------------------
# response_schemas = [
#     ResponseSchema(name="argument name", description="the name of the argument"),
#     ResponseSchema(name="argument value", description="The value of the argument extracted from the query. Don't write anything else here.")
# ]
# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

arg_extraction_prompt = PromptTemplate(template=TOOLS_PROMPT_EXAMPLES , 
                        input_variables=['arg_description','arg_dtype' ,'user_query'] ,   # ,'memory_examples'
                        # partial_variables= {"format_instructions" : output_parser.get_format_instructions()}
                        )

signature_chain = LLMChain(llm = small_llm, prompt = arg_extraction_prompt , verbose=False)


arg_filter_prompt = PromptTemplate(template=ARG_FILTER_PROMPT,
                                   input_variables=['query', 'arg_description'],
                                   partial_variables={"format_instructions": arg_filter_parser.get_format_instructions()}
                                   )
arg_filter_chain = LLMChain(llm = small_llm, prompt = arg_filter_prompt , verbose=False)
# new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)

#-------------------------------------------------------------------------------------------------------------------------------------------------

def fill_signature(query:str, arg_name:str , arg_dtype: dict , arg_descr :dict, tool_name:str)->Dict[str,Any] :
    if(len(query.strip('\n').strip().split()) == 1):
        return query
    extracted_args = signature_chain.run({'arg_description':arg_descr,'arg_dtype':arg_dtype, 'user_query':query})
    
    extracted_args =  extracted_args.strip('\n').strip(' ')
    extracted_args = re.sub(r'""', '"',extracted_args)
    

    if arg_dtype['argument_value'] == List[str]:
        if extracted_args[0] != '[':
            extracted_args = '['+extracted_args+']'

    if arg_dtype['argument_value'] == str:
        if extracted_args[0]=='[':
            extracted_args= extracted_args[1:-1]

    return extracted_args.strip('\n').strip(' ')

#-------------------------------------------------------------------------------------------------------------------------------------------------
def filter_arguments(query:str, arg_name , arg_descr :dict)->List[str] :
    argument_input = '\n'.join(['{name} : {descr}'.format(name = arg , descr = arg_descr[arg]) for arg in arg_name])
    response = arg_filter_chain.run({'query':query, 'arg_description':argument_input})
    x = None
    try : 
        output = arg_filter_parser.parse(response)
        print(output)
        x =  output['Arguments']
    except Exception as e:
        new_parser = OutputFixingParser.from_llm(parser=arg_filter_parser, llm=llm)
        output = new_parser.parse(response)
        print(output)
        x =  output['Arguments']

    final_args = []
    if type(x) is str:
        x = x.split(',')
    for arg in x:
        arg = arg.strip().strip('\n')
        if arg in arg_name:
            final_args.append(arg)

    return final_args




 