import json
import os
import pandas as pd
import tensorflow as tf
import subprocess
import json
import platform
import time
import openai
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from google.oauth2.credentials import Credentials
import re
from datetime import datetime, time, timedelta

import google.auth
from dateutil.parser import parse
from datetime import datetime
import datetime
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from dateutil import parser
from dateutil.parser import parse



def coding():
    # Set up the OpenAI API credentials
    openai.api_key = "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"

    prompt = "write a python program that uses tensorflow to train a machine learning model"
    prompt2 = "outline the structure of code needed to" + str(prompt) + "give it to me in the form of a list"
    # determine the different parts of the program
    # Define the model to use for generating the outline
    model = "text-davinci-002"

    # Call the OpenAI API to generate the outline
    response = openai.Completion.create(
        engine=model,
        prompt=prompt2,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.0,
    )

    # Extract the generated outline from the API response
    outline = response.choices[0].text.strip()

    # Print the generated outline
    print(outline)
    # for each part, make a call to openai to write the part of the code
    prompts = """
    import pandas as pd
    def preprocess(text):
        # TODO: Write preprocessing logic here
        return text
    model = build_model()
    train(model, data)
    """

    # Split the prompts string into a list of individual prompts
    prompt_list = re.split("\n|\r", prompts)
    # Loop through each prompt and use OpenAI's API to generate code
    for prompt in prompt_list:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # Extract the generated code from the OpenAI API response
        generated_code = response.choices[0].text.strip()
        
        # Print the generated code for each prompt
        print(f"Prompt: {prompt}\nGenerated code: {generated_code}\n")

        # create a directory that will hold the code
        # Define the name of the new directory
        new_dir = "Artificial_Intelligence\\Athena_Coding_Playground"

        # Create the new directory if it doesn't already exist
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            
        # Change the current working directory to the new directory
        os.chdir(new_dir)
        # save the code to a file
        with open("code.py", "w") as f:
            f.write(generated_code)
    # run the file
    # return the output
    #analyze the output, look for mistakes
    # if there are mistakes, make a call to openai to fix the mistakes
    # save the code to a file
    # run the file
    # return the output
    # Define the list of prompts (each prompt is on a separate line)

    
def main():
    coding()
if __name__ == '__main__':
    main()
