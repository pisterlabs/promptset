############### REGEXPLAINER ################
import os
import openai
import sys

def fetch_explanation(regex):
    openai.api_key = os.environ.get('OPENAI_KEY')

    # TODO ADD CONTEXT
    incipit_explanation = 'import re\nregex = r"' + regex + '"\n"""\nHere will be the step-by-step explanation of what types of patterns the regex is searching for:\n1.'
    response = openai.Completion.create(engine="davinci-codex", prompt=incipit_explanation, max_tokens=512, temperature=0.15)

    answer = "1." + response['choices'][0].text.split('"""')[0]

    lines = answer.split("\n")

    answer_non_rep = ""
    previous = ""
    
    for line in lines:
        # Checking after the numerotation
        if line[2:] != previous[2:]:
            answer_non_rep += "\n" + line
            previous = line
        else:
            break

    explanation =  "A quick explanation of the regex : " + regex + "\n" + answer_non_rep
    
    return explanation

