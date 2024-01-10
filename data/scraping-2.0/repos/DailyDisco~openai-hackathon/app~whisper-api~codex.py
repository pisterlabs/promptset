import os
from typing import List
import openai
import argparse # if __name__ == "__main__": let's us run the script from the command line
from dotenv import load_dotenv
import re

MAX_INPUT_LENGTH = 128

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def codexComplete(summary):
    # enriched_prompt = f"Transform this Python script into JavaScript:\n ### Python\n {summary}: \n### JavaScript"
    Platformresponse = openai.Completion.create(
        model="code-davinci-002",
        prompt = f"Transform this Python script into JavaScript:\n ### Python\n {summary}: \n### JavaScript".format(summary),
        # prompt="Write a console log of this: {}".format(summary),
        temperature=0,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["###"]
    )
    
    print(f"Snippet: {Platformresponse}")
    return Platformresponse.choices[0].text