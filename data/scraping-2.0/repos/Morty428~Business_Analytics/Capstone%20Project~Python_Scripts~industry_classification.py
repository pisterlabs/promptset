"""
Created for Neuro Insights

@author: Matthew Mortensen
@email: matthew.m.mortensen@gmail.com

parts of the code have been sanitized due to NDA
"""
import openai
# This is not the best way to impliment the openai key
secret_key = "openai-key"

openai.api_key = secret_key


#function to use davinci-003 from openai to provide a company name and get the industry category and subcategory
#openai needs api key defined to operate, each request costs money. refer to openai pricing for details
#input needed; string or string array containing only the company name
def openai_categorizer(name: str):
    #prompt to generate completions-uses few shot encoding to help refine prompts
    #prompt is designed to try and limit hallucinations and give a desired output
    #prompt is designed to give an answer of 'Unknown' if a name is passed that is not a company
    #'Unknown' responses can be reduced by fine-tuning
    #see openai documentation for fine tuning instructions
    
    #prompt has been sanitized 
    f_prompt = '''
    '''
    #sub prompt
    f_sub_prompt = "{name}"
    prompt = f_prompt.format(name=name) #add name to prompt
    sub_prompt = f_sub_prompt.format(name=name)
    print(sub_prompt)
       
    response = openai.Completion.create(model="text-davinci-003",
        prompt=prompt, #prompt given to generate responses
        temperature=0, #between 0 and 2. Higher values will make the output more random, lower values more deterministic.
        max_tokens=64, #maximum number of tokens to generate in the completion similar to max number of words
        #top_p=1.0, #alternative to temperature generally recommend altering top_p or temperature but not both
            #0.1 means only the tokens comprising the top 10% probability mass are considered.
        #Number between -2.0 and 2.0. Positive values penalize new tokens decreasing likelihood to repeat the same line verbatim
        frequency_penalty=0.0, #
        #Number between -2.0 and 2.0. Positive values penalize new tokens increasing likelihood to talk about new topics.
        presence_penalty=0.0,
        #if subcategories have multiple responses, the stop sequence paramerter prevents multiple subcategories
        stop=[',']
    )
    response_txt = response['choices'][0]['text']
    #output is string or string array of category and subcatecory
    
    return response_txt