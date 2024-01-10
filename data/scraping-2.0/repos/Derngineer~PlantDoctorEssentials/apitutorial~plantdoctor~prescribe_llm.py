from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

print('THIS FILE IS LOADING FULL TIME')

load_dotenv ()
open_api_key = os.getenv("api_key")
if open_api_key == None:
    print("we dont have a working API key")
else:
    print("yes key picked", open_api_key)



llm = OpenAI(openai_api_key=open_api_key, temperature = 1)

print('this file is coming through .....')

def create_prescription(name, disease):

    print('predict function is on')
    name = name
    disease = disease


    input_variables = ['plant_name','plant_disease']
    template = '''You are a professional agriculturalist with a PHD in horticulture \
    and plant disease detection, you have various tools and resource.You job is to\
    provide a detailed, easy to understand prescription for a plant named {plant_name} with\
    the disease named {plant_disease}. Make sure your prescription is professional and as\
    accurate as possible. Specify any fungicides, pesticides, herbicides if the problem needs any\
    , best practices that may\
    Give out as much information as possible. Dont hesitat to give details on related solutions\
    be used to combat the problem. In addition, at the bottom specify any precautions to be taken, more\
    so let the user known that they are still advised to seek professional advise, make it clear the entire\
    system was created using an AI system. As for the presentation, your prescription should be in a proffessional\
    formal with bullet points when ever there is a need for them, make sure you skip lines to make everything clear\
    please bring out the answer with a pre-formatted HTML format, with highlights to key ponts and line gaps, bulletins etc\
    If the plant is healthy ie plant_disease is healthy, congratulate the user and encourage them to nourish their plants\
    in addition tell them how they can maintain their plant's health.\
    
    formatting instructions are as follows\
    here are tags you should use for listing <ul><li></li></ul>, heading <h3></h3>, to skip line <br>\
    to put a horizontal line <hr>, to make bold use <strong><\strong> \
    For each key field skip please skip lines . skip lines whenever needed\
    Provide more details on the subject diseases as much detail as possible\
    Please provide a well detailed neat structure\
    '''

    prompt = PromptTemplate(input_variables = input_variables, template= template)

    # prompt.format(plant_name = name, plant_disease = disease)
    variable_dictionary = {
        'plant_name':name,
        'plant_disease': disease,
    }

    chain = LLMChain(llm =llm, prompt = prompt)

    prescription = chain.run(variable_dictionary)

    return prescription


# print(create_prescription('potato', 'potato_leaf_blight', "thats the printed prescriptionn"))