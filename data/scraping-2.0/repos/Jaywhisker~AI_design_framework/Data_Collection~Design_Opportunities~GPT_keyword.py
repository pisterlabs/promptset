#####################################################################################################
# This is the functions to retrieve the product specification, flaws and strengths from GPT-3
# imports: openai, ast and helper functions from Helper.

######################################################################################################

import openai
import ast
import nltk
from ...Helper import *

#function to generate the product specification
def get_specifications(search_terms, model, apikey):
    openai.api_key = apikey
    prompt = "give me the design specifications of the" + " ".join(search_terms) + "in a python dictionary in the form of specifications:value."
    output = generate_texts(prompt, model)
    #sometimes the output may be output = {}, if so remove output =
    try:
        index = output.index("{")
        output = output[index:].strip()
    except:
        pass
      
    design_specifications = ast.literal_eval(output) #remove the string as output = '{}' such that it is a dictionary
    print("design specifications", design_specifications)
    return design_specifications



#edited ver of get_specification for product flaws
def get_flaws(search_terms, model, apikey):
    openai.api_key = apikey
    prompt =  "give me the design flaws of the" + " ".join(search_terms) + "in a python dictionary in the form of specifications:value."
    output = generate_texts(prompt, model)
    try:
        index = output.index("{")
        output = output[index:].strip()
    except:
        pass

    design_flaws = ast.literal_eval(output)
    print("design flaws", design_flaws)
    return design_flaws


#edited ver of get_specification for product strengths
def get_strength(search_terms, model, apikey):
    openai.api_key = apikey
    prompt =  "give me the design strengths of the" + " ".join(search_terms) + "in a python dictionary in the form of specifications:value."
    output = generate_texts(prompt, model)
    try:
        index = output.index("{")
        output = output[index:].strip()
    except:
        pass
    design_strengths = ast.literal_eval(output)
    print("design strengths", design_strengths)
    return design_strengths

#edited ver of get_competitors for product
def get_competitors(search_terms, model, apikey):
    openai.api_key = apikey
    prompt =  "give me the competitors of the" + " ".join(search_terms) + "in a python dictionary in the form of product:company."
    output = generate_texts(prompt, model)
    try:
        index = output.index("{")
        output = output[index:].strip()
    except:
        pass
    competitors = ast.literal_eval(output)
    print("competitors", competitors)
    return competitors


