#!/usr/bin/env python
# coding: utf-8

# All imports
import streamlit as st
from langchain.chains import create_extraction_chain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

import pandas as pd
from typing import Any, List, Optional
import warnings
warnings.filterwarnings("ignore")


@st.cache_data(show_spinner = False)
def create_schema_from_excel(orignal_excel_file, llm_prompts_index):
    '''
    create_schema_from_excel: This function will automatically create a schema based on the "Field Name" and "Data Type" columns in results_output.xlsx

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    llm_prompts_index: List of the index of the rows of prompts (in orignal_excel_file) that were fed to LLM 

    Output - None
    '''

    filtered_df = orignal_excel_file.iloc[llm_prompts_index]

    schema = {"properties": {}}

    for index, row in filtered_df.iterrows():

        field_name = row["Field name"]
        data_type = row["Data Type"]

        property_dict = {"type": data_type}
        
        schema["properties"][field_name] = property_dict
        
    # Return the schema generated
    return schema 


# @st.cache_data(show_spinner = False)
def _get_extraction_function(entity_schema: dict) -> dict:
    '''
    _get_extraction_function: This is the information_extraction function returning a dictionary

    Input - 
    entity_schema: Takes the entity schema dictionary as input

    Output - 
    Below dictionary
    '''
    
    return {
        "name": "information_extraction",
        "description": "Extracts the relevant information from the passage.",
        "parameters": {
            "type": "object",
            "properties": {
                "info": {"type": "array", "items": _convert_schema(entity_schema)} # calling _convert_schema function from langchain 
            },
            "required": ["info"],
        },
    }

# @st.cache_data(show_spinner = False)
# @st.cache_resource(show_spinner = False)
def create_extraction_chain(
    schema: dict,
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    verbose: bool = False,
) -> Chain:
    
    """
    Create_extraction_chain: Creates a chain that extracts information from a passage.

    Input - 
        schema: The schema of the entities to extract.
        llm: The language model to use.
        prompt: The prompt to use for extraction.
        verbose: Whether to run in verbose mode. In verbose mode, some intermediate
            logs will be printed to the console. Defaults to the global `verbose` value,
            accessible via `langchain.globals.get_verbose()`.

    Output - 
        Chain that can be used to extract information from a passage.
    """
    
    # Call _get_extraction_function which returns a dictionary 
    function = _get_extraction_function(schema)

    # Extraction template that user enters
    # Note: recommended you keep here 'Passage: {input}' and the extraction template as follows as well

    _EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned\
    in the following passage together with their properties.

    Only extract the properties mentioned in the 'information_extraction' function.

    If a property is not present and is not required in the function parameters, do not include it in the output.

    If output is a Date then change it to dd/mm/yyyy format.

    Passage:
    {input}
    """ 
        
    extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    
    output_parser = JsonKeyOutputFunctionsParser(key_name="info")
    
    llm_kwargs = get_llm_kwargs(function)
    
    # Construct the LLMChain
    chain = LLMChain(
        llm=llm,
        prompt = extraction_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        verbose=verbose,
    )
    
    # Return the chain
    return chain

# @st.cache_data(show_spinner = False)
# @st.cache_resource(show_spinner = False)
def parse_value(output_response, llm_prompts_index, orignal_excel_file, schema, llm):
    '''
    parse_value: This function will take the values in column, "Output Result", feed it to parser and generate a list of key-value pairs in the dictionary

    Input - 
    output_response: List containing response of prompts passed to LLM
    llm_prompts_index: List of the index of the rows of prompts (in orignal_excel_file) that were fed to LLM 

    Output - 
    orignal_excel_file: Dataframe of the results excel file
    '''

    final_output_value = []

    for output_value in output_response:
        
        # Create chain
        chain = create_extraction_chain(schema, llm)
        chain_result = chain.run(output_value)
        final_output_value.append(chain_result)

    print(llm_prompts_index)
    print(final_output_value)

    # Ensure that the "Final Output result" column accepts object (dictionary) data type
    orignal_excel_file['Final Output result'] = None  # Initialize the column with None values
    orignal_excel_file['Final Output result'] = orignal_excel_file['Final Output result'].astype(object)
    
    # Iterate through llm_prompts_index and assign values from final_output_value to a new column, "Final Output result"
    for index, info_dict in zip(llm_prompts_index, final_output_value):
        orignal_excel_file.at[index, 'Final Output result'] = info_dict
    
    return orignal_excel_file


