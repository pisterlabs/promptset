import openai
from openai import OpenAI
import json
import os

# openai.api_base = 'https://api.deepinfra.com/v1/openai'

if 'config.json' in os.listdir('./docs/'):
    with open('./docs/config.json') as config_file:
        config = json.load(config_file)
        api_key = config['api-key']
else:
    api_key = os.environ.get('API_KEY')

client = OpenAI(
    api_key=api_key   
)

# resume_temp = resume.replace('\n','')

def send_prompt_instruct(system_prompt, prompt):
    MODEL_DI = "gpt-3.5-turbo-instruct"
    response = client.completions.create(
        model=MODEL_DI, # Optional (user controls the default)
        # messages=[
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": prompt},
        #     # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}
        # ],
        prompt = f'''
{system_prompt}

{prompt}''',
        seed = 7,
        temperature = 0, #Randomness
        max_tokens = 1000, #Maximum words
        top_p = 0.1,
    )

    reply = response.choices[0]
    print(reply.text)

    return reply.text

def send_prompt_chat(system_prompt, prompt):
    MODEL_DI = "gpt-3.5-turbo-1106"
    response = client.completions.create(
        model=MODEL_DI, # Optional (user controls the default)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}
        ],
        seed = 7,
        temperature = 0.5, #Randomness
        max_tokens = 1000, #Maximum words
        top_p = 0.1,
    )

    reply = response.choices[0]
    print(reply.text)

    return reply.text

def jd_prompt_1(jd_text):
    prompt = f'''
    Job Description:
    {jd_text}'''

    system_prompt = f'''
    Strictly extract the all skills relevant to this job description. Group them under Technical Skills, Soft Skills and Languages in this Format, only include the language name (e.g. English, Chinese, German):
    Job name: ...
    Technical Skills: ...
    Soft Skills: ...
    Languages: ...
    
    Do not include any languages in the soft skills section. Remove brackets in the language section.
    '''
    
    reply = send_prompt_instruct(system_prompt, prompt)
    return reply

def resume_prompt(resume_text):
    prompt = f'''
    Resume:
    {resume_text}'''

    system_prompt = f'''
    Strictly extract name (without honorifics) of the applicant, and all possible and relevant skills, then group the skills in this format: 
    Name: ...
    Technical Skills: ... 
    Soft Skills: ... 
    Languages: ...
    '''
    
    reply = send_prompt_instruct(system_prompt, prompt)
    return reply
    
   
#Function to convert LLM Output to dictionary
def convert_to_dict(text):
    test = text.replace('\n-',',')
    test = test.replace(": ,",':')
    test = test.replace("\n\n",'\n')
    # Get result from Name onwards
    if 'Job name' in test:
        test = test[test.find('Job name'):]
    else:
        test = test[test.find('Name'):]

    # If chatbot writes a note, get result before the note
    if 'Note' in test:
        test = test[:test.find('Note')]

    # Remove whitespace
    test = test.strip()
    # Split by lines
    test = test.split('\n')

    print('-'*10 + 'Before Dictionary'+ '-'*10)
    print(test)

    #Initialize dictionary
    outputdictionary = {}

    for i in test:
        # Get variable after the first instance of ":"
        templist = i.split(':',1)
        if len(templist) <= 1:
            pass
        else:
            # If there is more than 1 element in the variable
            if len(templist[1].split(',')) > 1:
                # Get a list of strings in the output variable
                stringifylist = templist[1].split(',')

                # Initialize new list
                newlist = []

                # Remove white spaces for each string in the list
                for string in stringifylist:
                    newstring = f'{string.strip()}'
                    newlist.append(newstring)
                # S
                templist[1] = newlist
                outputdictionary[templist[0]] = templist[1]
            else:
                outputdictionary[templist[0]] = templist[1].strip()

    print('-'*10 + 'Dictionary' + '-'*10)
    print(outputdictionary)
    return outputdictionary

def chatbot():
    return client