'''

Goal:
    convert the question in natural language into a structured format that will be used by the math solver

Input:
    question in natural language
    model - the model to use to generate the answer. Default is gpt-4-0314

Output:
    structured format of the question
    total tokens used

Example:
    Input:  "If a car is traveling at 60 miles per hour, how many miles will it travel in 2 hours?"
    Output: "<start>desired_variable=distance_travelled;miles_per_hour=60;time_in_hours=2;distance_travelled=miles_per_hour*time_in_hours<stop>"

'''

import sys
import os
import openai
from config import openai_key

#set openai api key
openai.api_key = openai_key

def condenseQuestion(question,model="gpt-4-0314"):
    #define the prompt
    pre_prompt = '''
    I am about to provide a question in natural language and I want you to isolate the variables and their values, and any relevant formulas. 
    You response will be the only info given to the next model, so be as explicit and consistent as possible.
    If it is not known, just say variable=?? or provide the formula to calculate it. 
    Include any formulas that are implicitly defined in the question, but only used variables that are already defined.
    Once a variable is used, be sure to reference it by the same name throughout the question. For example, do not refer to a variable as "time" and then later as "t".
    Only include 1 equal sign per formula.
    Include all units, with a space between the value and the unit. Do not include any other spaces in the formula.
    Do not include any numbers in the variable names. ie do not say x1 say x_one.
    Provide this in plain csv format with a start and stop token. 
    Explicitly state which variable is being asked to solve for, like this: desired_variable=variable_one. Do not use any numbers in the variable name.
    Be as explicit as possible, do not abbreviate any variable name. For example, instead of saying area, say area_of_square.
    Do not define anything in the variable name what could be a formula. For ex, instead of saying time_when_x_equals_a=??, say time=?? and then include x=a as another formula.
    Only provide the csv string, no need for anything else.
    example: "<start>desired_variable=width_of_square;length_of_square=5;base_area_of_pyramid=7;width_of_square=??<stop>". 
    another example: "<start>desired_variable=total_meters_run_per_week;sprints_per_day=3;days_per_week=3;distance_per_sprint=60;total_meters_run_per_week=distance_per_sprint*days_per_week*sprints_per_day<stop>"
    ok, here is the question: 
    '''
    s_prompt = pre_prompt + '"' + question + '"'

    def get_response(s_model,message):
        '''
        send request to openai to generate the answer
        '''
        response = openai.chat.completions.create(
            model = s_model,
            temperature = 1,
            messages = [
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content, response.usage.total_tokens

    # generate the answer
    try:
        answer,total_tokens = get_response(model,s_prompt)
    except:
        raise Exception(f"Error generating answer. Please confirm that you have access to {model} through your openai api. You can change the model used in src/condense.py.")

    #remove \n from answer
    answer = answer.replace("\n", "")
    return answer, total_tokens

if __name__ == "__main__":
    question = "If a car is traveling at 60 miles per hour, how many miles will it travel in 2 hours?"
    answer, total_tokens = condenseQuestion(question)
    print(answer)