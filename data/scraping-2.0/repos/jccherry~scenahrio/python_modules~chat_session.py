# chat_session.py

import os
import openai
import ast

openai_api_model = 'gpt-4'
#openai_api_model = 'gpt-3.5-turbo'

system_prompt = f"You are HR-GPT, a Human Resources Simulation AI. Your role is to provide HR users with predictions and suggestions " \
                f"on how to respond to potential recruitment targets or employees. Responses from HR will be labeled 'HR' and responses " \
                f"from others will be labeled with their name."

# from a details dictionary, generate input for the openAI API
# returns a tuple with messages for the api and direction on who the
# next user in the chain is
def generate_chat_from_details(details, num_responses = 3, num_words = 20):
    print("generate_chat_from_details called!")

    print(details)

    # Create a blank array which will contain our messages
    messages = []

    # add in the system prompt
    messages.append({'role' : 'system', 'content' : system_prompt})

    # Add in some content about the employee
    messages.append({'role' : 'system' , 'content' : f"""
Employee Profile:
```
Name: {details['profile_name']}
Age: {details['profile_age']}
Gender: {details['profile_gender']}
Job Title: {details['profile_job_title']}
Salary: {details['profile_salary']}
Notes: {details['profile_notes']}
```"""})
    # some context about the scenario:
    messages.append({ 'role' : 'system', 'content' : f"Context: {details['context']}"})
    
    last_user_spoke = 'HR'

    for message in details['messages']:
        last_user_spoke = message['user']
        messages.append({
            'role' : 'user'
            , 'content' : f"{message['user']}: '{message['message']}'"
        })
    
    next_user_spoke = details['profile_name'] if last_user_spoke == 'HR' else 'HR'

    # Guidance at the end to make the system give good output
    messages.append({ 'role' : 'system', 
                     'content' : f"""HR's desired outcome is to: {details['desired_outcome']}
Use knowledge about the employee's profile, context, and HR's desired outcome to simulate realistic possible responses.
Generate {num_responses} possible responses from {next_user_spoke}, each with {num_words} words or less.
Vary the responses' sentiment from positive to negative.
Output must be parse-able with python\'s ast.literal_eval() and nothing else, such as ["Let\'s go.", "Don\'t fire me.", "I\'m excited!"].
"""})

    return (messages, next_user_spoke)

def generate_responses_from_chat(chat, next_user):
    openai.api_key = os.getenv('OPENAI_API_KEY')

    #print('generate_responses_from_chat called!')
    #print(f'os.getenv(OPENAI_API_KEY) = {os.getenv("OPENAI_API_KEY")}')
    #print(chat)

    response = openai.ChatCompletion.create(model=openai_api_model, messages=chat)
    response_content = response['choices'][0]['message']['content']

    try:
        response_list = ast.literal_eval(response_content)
    except SyntaxError:
        print("Attempting to fix a syntax error in GPT response...")
        # If there is an issue parsing the string, attempt to have GPT repair it:
        fix_messages = [
            {"role": "system", "content": "You are AST-GPT, a syntax repair bot that takes input in the form of a string and repairs it so that it is parseable with python's ast.literal_eval.  In your response, include only fixed strings with no additional context."}  # Initial system prompt
            , {"role": "user", "content": list_string}  # broken string
        ]

        for i in range(3):
            response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=fix_messages)
            list_string = response['choices'][0]['message']['content']
            try:
                response_list = ast.literal_eval(list_string)
                break
            except SyntaxError:
                pass

    print(f"Responses: {type(response_list)}")
    print(response_list)

    # take the items in the response list and turn it into nice object structures
    children = []
    for response in response_list:
        children.append({
            'user' : next_user
            , 'message' : response
        })

    return children