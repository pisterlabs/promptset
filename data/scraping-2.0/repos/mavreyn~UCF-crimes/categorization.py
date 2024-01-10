'''
Using OpenAI to categorize incidents
'''

import openai
import pandas as pd

CATEGORY_DICT = {
    1: "Theft and Property Offenses",
    2: "Drug-Related Offenses",
    3: "Assault and Battery",
    4: "Traffic Offenses",
    5: "Stalking and Harassment",
    6: "Fraud and White-Collar Crimes",
    7: "Weapons and Firearm Offenses",
    8: "Miscellaneous Offenses"
}

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    txt = response.choices[0].message["content"]
    print(txt)
    return txt

def make_prompt(title):
    return f"""You are a model helping to classify incidents that occur on a local college campus. The team we are working with has divided the incidents into 8 categories:

    1. Theft and Property Offenses
    2. Drug-Related Offenses
    3. Assault and Battery
    4. Traffic Offenses
    5. Stalking and Harassment
    6. Fraud and White-Collar Crimes
    7. Weapons and Firearm Offenses
    8. Miscellaneous Offenses

    Please classify the following incident (between triple backticks) into one of the above categories by outputting a single character ranging from 1-8. If you are unsure, output '8' (Miscellaneous Offenses).

    ```{title}```"""

def main():
    crimes = [] # LIST OF INCIDENT TITLES

    for crime in crimes:
        prompt = make_prompt(crime)
        category = CATEGORY_DICT[int(get_completion(prompt))]
        print(category)
        # ADD TO DATAFRAME