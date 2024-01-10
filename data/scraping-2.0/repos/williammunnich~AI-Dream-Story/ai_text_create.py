# Importing necessary libraries
import os
import openai
import random
from dotenv import load_dotenv

# Defining function to return text and choice
def return_text_and_choice():
    # Loading environment variables from .env file
    load_dotenv()
    # Setting API key for OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Defining upper and lower limits for random year
    upper_limit_rand = 3000
    lower_limit_rand = -10000
    year = None
    # Generating a random year between lower and upper limits
    rand_number = random.randint(lower_limit_rand, upper_limit_rand)
    if rand_number <= -1:
        rand_number = rand_number * -1
        rand_number = str(rand_number)
        year = rand_number + " BC"
    else:
        rand_number = str(rand_number)
        year = rand_number + " AD"
    # Defining helper function to read a file
    def read_file(filename):
        with open(filename, 'r', encoding='UTF8') as file:
            data = file.read().replace('\n', '')
            return data


    # Formulating a request to OpenAI API
    formulated_request = "You wake up in your bed on a normal day in the year " + year + \
        ".\nWrite a few lines about what you do for the day then provide two choices for how to continue the day \
        in the form of 'Choice 1:' and 'Choice 2:'"

    # Sending request to OpenAI API and storing the response
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt = formulated_request,
    temperature=0.39,
    max_tokens=500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    # Storing the output and returning
    output = "The year is ", year, " .", response['choices'][0]['text']
    return str(output)
    
# Defining function to return text and choice given previous text and option
def given_input_return_text_and_choice(previous_text, option):
    # Loading environment variables from .env file
    load_dotenv()
    # Setting API key for OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Defining helper function to read a file
    def read_file(filename):
        with open(filename, 'r', encoding='UTF8') as file:
            data = file.read().replace('\n', '')
            return data

    # Formulating a request to OpenAI API
    formulated_request = "Given that this is what happened previously: " + previous_text + " and the option that was\
        chosen was " + option + " continue the story onwards and provide exactly two options on what to do next."
    # Call OpenAI API to get the response
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt = formulated_request,
    temperature=0.39,
    max_tokens=500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    # Format the output
    output = response['choices'][0]['text']
    
    #return output
    return str(output)  
  
print(return_text_and_choice())