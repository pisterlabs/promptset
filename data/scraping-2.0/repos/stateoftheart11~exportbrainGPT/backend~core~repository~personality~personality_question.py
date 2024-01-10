import os
import openai
import json
import re


responsiveness = ["Disagree Strongly", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree strongly"]
traits = ['extraversion', 'neuroticism', 'conscientiousness']

prompt_text_muliple = """Generate {question_number} question to evaluate "{trait}" in a personality test. 
The options for the respondent are: 
1. {responsiveness[0]} 
2. {responsiveness[1]} 
3. {responsiveness[2]} 
4. {responsiveness[3]} 
5. {responsiveness[4]} 

<<<INSTRUCTION>>>
Here are some rules that the generated <response> should follow.
If the positive answer refers to {trait}, the <response> should be formated follow.

##### POSITIVE #####
<<QUESTION>>
####################

Else, the question should be formated following.
##### NEGATIVE #####
<<QUESTION>>
####################

Response: <response>
"""

# Muliple questions
def generate_question(trait='extraversion', question_number:int=1):
    openai.api_key = os.getenv("OPENAI_API_KEY")  
    output_list = []
    # for i in range(2):  # change this to a higher number if more questions are desired for each trait
    
    prompt = prompt_text_muliple.replace("{question_number}", str(question_number)).replace("{trait}", trait)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.8,
        max_tokens=100
    )

    res = response.choices[0].text.strip()
    matches = re.findall("##### (.*?) #####\n(.*?)\n####################", res, re.DOTALL)
    output_list = output_list + [{
        "trait": trait,
        "positive": match[0].upper() == "POSITIVE",
        "question": match[1].strip()
    } for match in matches]
    return output_list

def generate_question_all(question_number:int=1):
    output_list = []
    for trait in traits:
        output_list = output_list + generate_question(trait, question_number=question_number)
    return output_list
    # return json.dump(output_list)
