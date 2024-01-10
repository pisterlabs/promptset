import os
import openai
import json
import re
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # 'your-api-key'  # replace 'your-api-key' with your actual key

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
    # questions.append(question)
    # questions.append(question)
    return output_list

def generate_question_all(question_number:int=1):
    output_list = []
    for trait in traits:
        output_list = output_list + generate_question(trait, question_number=question_number)
    return json.dump(output_list)

## single question

# prompt_text = """Generate a question to evaluate "{trait}" in a personality test. 
# The options for the respondent are: 
# 1. {responsiveness[0]} 
# 2. {responsiveness[1]} 
# 3. {responsiveness[2]} 
# 4. {responsiveness[3]} 
# 5. {responsiveness[4]} 

# <<<INSTRUCTION>>>
# Here are some rules that the generated <response> should follow.
# If the positive answer refers to {trait}, the <response> should be formated follow.

# ##### POSITIVE #####
# <<QUESTION>>
# ####################

# Else, the question should be formated following.
# ##### NEGATIVE #####
# <<QUESTION>>
# ####################

# Response: <response>
# """

# def generate_question(trait='Extraversion', question_number=1):
#     output_list = []
#     for _ in range(question_number):  # change this to a higher number if more questions are desired for each trait
#         prompt = prompt_text.replace("{trait}", trait)
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=prompt,
#             temperature=0.8,
#             max_tokens=100
#         )

#         res = response.choices[0].text.strip()
#         matches = re.findall("##### (.*?) #####\n(.*?)\n####################", res, re.DOTALL)
#         output_list = output_list + [{
#             "trait": trait,
#             "positive": match[0].upper() == "POSITIVE",
#             "question": match[1].strip()
#         } for match in matches]
#         # questions.append(question)
#     return output_list

if __name__ == '__main__':
    # json_info = []
    for trait in traits:
        output_list = generate_question(trait, question_number=4)
        print(output_list)


"""
Now I am going to assess personality in extraversion, Neuroticism, conscientiousness

Users will answer in multiple choice question answering, in other words, they can choice one option in following 7 options:

1. disagree strongly
2. disagree a little
3. neither agree nor disagree
4. agree a little
5. agree strongly

I need test questions for testing

I am going to use openai gpt model and generate questions using prompt.
Give me python code.

[
{
    "trait": "extraversion"
    "question": "I love being the center of attention."
},
{
    "trait": "neuroticism"
    "question": "I change my mood a lot."
},
{
    "trait": "conscientiousness"
    "question": "I am exacting in my work."
},
{
    "trait": "extraversion"
    "question": "I feel comfortable around people."
},
{
    "trait": "neuroticism"
    "question": "I often feel blue."
},
{
    "trait": "conscientiousness"
    "question": "I am always prepared."
}
]
"""

# [{
#     "positive": True,
#     "question": "Do you enjoy being around people?"
# },
# {
#     "positive": False,
#     "question": "Do you prefer to be alone?"
# }]
