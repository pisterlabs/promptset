import flask 
from flask import session
from flask_session import Session
from waitress import serve
from flask import request, jsonify, render_template, request, url_for, redirect

# Importing the required libraries
from dotenv import load_dotenv, find_dotenv
import os
import tiktoken
import sys
import utils
import openai

"""LINK TO BACKEND: http://localhost:8000/"""
# Getting the OpenAI API Key
openai.api_key  = "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"
global history
history = []
# This function auto completes GPT's reponse
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
    )
    return response.choices[0].message["content"]

# Main function to repond to the user and moderate question and response
def process_user_message(user_input, all_messages, debug=True):
    delimiter = "####"
    user_input = user_input.replace(delimiter, "")
    
    # Step 1: Check input to see if it flags the Moderation API or is a prompt injection
    response = openai.Moderation.create(input=user_input)
    moderation_output = response["results"][0]

    if moderation_output["flagged"]:
        print("Step 1: Input flagged by Moderation API.")
        return "Response to user: Sorry, we cannot process this request. Your request was flagged by our moderation checks.", all_messages

    if debug: print("Step 1: Input passed moderation check.")

    # Step 2: Answer the user question
    system_message = f"""
    Follow these steps to answer the patient queries.
    The patient query will be delimited with four hashtags,\
    i.e. {delimiter}. 

    Step 1:#### First categorize the user's query into one \
    of the following categories: (1) The user is asking for general information about a \
    condition. (2) The user is listing symptoms and wants information \
    on what is wrong with them. (3) The user wants information on a type of doctor and what they do. \
    (4) The user wants to know more about a mental health related issue \
    and what they can do. (5) The user is not asking about anything medical related do not categorize \
    the query.

    Step 2:#### Based on how the user's query was categorized above do \
    the following based on the category number: (1) List common symptoms of the condition, state how to resolve the \
    issue and how to prevent it. (2) If very little information was given about their symptoms, \
    then ask more follow up questions to narrow down that is wrong. Make sure to ask the user relevant follow-up questions. \
    Give them specific prompts and ask specific questions. (5) If the user is greeting you, \
    answer their question and respond back. Otherwise tell the user that you cannot help them with this query and \
    let them know that you can help with anything medical related

    Step 3:#### If the query was placed in catergory (2) and you are sure of the patient's condition, \
    then decide wether the user is in critical condition or not. If they are in critical \
    condition tell the user how they can minimize risks immediately \
    and to contact a local doctor. If they are able to treat \
    the issue themselves and are not in critical condition, \
    then tell them how to treat the issue. Answer \
    to the patient in a friendly and helpful tone. 
    
    Remember to always have a reponse to user!

    Use the following format:
    Step 1:#### <step 1 reasoning>
    Step 2:#### <step 2 reasoning>
    Step 3:#### <step 3 reasoning>
    Response to user:#### <response to customer>

    Make sure to include {delimiter} to separate every step.
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"}
    ]

    final_response = get_completion_from_messages(all_messages + messages)
    if debug:print("Step 2: Generated response to user question.")
    all_messages = all_messages + messages[1:]

    # Step 3: Put the answer through the Moderation API
    response = openai.Moderation.create(input=final_response)
    moderation_output = response["results"][0]

    if moderation_output["flagged"]:
        if debug: print("Step 3: Response flagged by Moderation API.")
        return "Response to user: Sorry, we cannot provide this information.", all_messages

    if debug: print("Step 3: Response passed moderation check.")

    return final_response, all_messages

# Function to collect all messages
def collect_messages(user_input, debug=False):
    global history
    print(f"\nUser Input = {user_input}")
    if user_input == "":
        return ""
    
    response, context = process_user_message(user_input, history, debug=False)
    print(response)

    context.append({'role':'assistant', 'content':f"{response}"})
    history = context
    return response


app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'

@app.route('/', methods=['GET'])
def home():
    return '''<h1>home page</h1>
'''

@app.route('/gpt', methods=['GET'])
def returnQuery():
    global history

    question = request.args["query"]
    question = question.replace("_", " ")
    answer = collect_messages(question)
    if '####' in answer:
        answer = answer.split('###')[-1].strip()
    else:
        answer = answer.split('\n')[-1].strip()[17:]
    answer = jsonify(answer)
    
    answer.headers.add("Access-Control-Allow-Origin", "*")
    return answer

serve(app, host='0.0.0.0', port=8000)