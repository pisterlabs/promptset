from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from typing import Dict

# Authentify with OpenAI key

env_path = Path('/content/ATT85165.env')
load_dotenv(dotenv_path=env_path)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print("environement is build")

# Build function to extract fields from txt


def extract_fields(extracted_texts: Dict[str, str]) -> pd.DataFrame:

    # authentificate
    env_path = Path('/content/ATT85165.env')
    load_dotenv(dotenv_path=env_path)

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    print("environement is build")

    # Fields to target and retrieve as columns
    COLUMNS = ['ISIN', 'Issuer', 'Ccy',
               'Underlying(s)', 'Strike', 'Launch Date', 'Final Valuation Day', 'Maturity', 'Cap', 'Barrier']
    # prompting category
    prompting_category = {
        "ISIN": {
            "Rule": "International Securities Identification Number from the term sheet. The ISIN is a 12-character alphanumeric code that uniquely identifies the bond for accurate tracking and trading. Look for the word 'ISIN' in close proximity, and ensure that the extracted string is exactly 12 characters. It can also be a CUSIP number of 9 characters.",
            "Restriction": "It can't be a float."
        },
        "Issuer": {
            "Rule": "The entity or organization that issues the financial product. Identify and extract the issuer's name (both full name and initials) from the term sheet. In the provided data, examples include RBC, BNP, Citi, MS, and GS.",
            "Restriction": "It can't be a number."
        },
        "Ccy": {
            "Rule": "Extract the currency mentioned in the term sheet. The currency in which the financial product is denominated or traded. Check that the currency exists in a specified list; currency can look like: EUR, USD in the provided data. They can be multiple.",
            "Restriction": "It can't be a number."
        },
        "Underlying(s)": {
            "Rule": "The assets, indices, or securities on which the financial product's performance is based, for example: GLE, RNO FP, VOW3 GY, DAI GY. It can also be called bloomberg code/identifier. Keep in mind that if multiple underlyings are present then each underlying has its unique strike price.",
            "Restriction": "It can't be a number"
        },
        "Strike": {
            "Rule": "The strike price of the underlying and not the barrier. It comes as a percentage attached to automatic early redemption. You can also calculate it by taking the (knock-in barrier price divided by barrier level). It can also be called ETI Initial.",
            "Restriction": "It cannot be 100%, It can't contain letters."
        },
        "Launch Date": {
            "Rule": "Find and extract the launch date (Trade Date) from the term sheet. The launch date marks the initiation of the bond. Provide it in the format dd/mm/yyyy and verify its accuracy.",
            "Restriction": "It can't be in another format that is not dd/mm/yyyy "
        },
        "Final Valuation Day": {
            "Rule": "Identify and extract the final valuation day from the term sheet. The final valuation day is crucial for determining the bond's concluding value. Provide it in the format dd/mm/yyyy and ensure precision.",
            "Restriction": "It can't be in another format that is not dd/mm/yyyy "
        },
        "Maturity": {
            "Rule": "Extract the maturity date from the term sheet. The maturity date signifies when the bond reaches maturity. Provide it in the format dd/mm/yyyy and verify its accuracy.",
            "Restriction": "It can't be in another format that is not dd/mm/yyyy"
        },
        "Cap": {
            "Rule": "Find and extract the cap value mentioned in the term sheet. The cap value influences the maximum value of a variable and impacts the bond's potential returns. Confirm the accuracy of the cap values and check for associated conditions.",
            "Restriction": "It can't be a number or a letter"
        },
        "Barrier": {
            "Rule": "The barrier is a percentage. It is a specific level or threshold that, if reached, can trigger certain events or determine the product's performance. Look for the word 'Barrier' in close proximity.",
            "Restriction": "It can't be a number that is not a percentage."
        }
    }

    # prompt
    prompt = f''''
    
    You are the best financial data analyst in the world with special expertise on term sheets.

    
    Your task is to go through these texts files {extracted_texts} and extract the value of the corresponding key elements listed here {COLUMNS}.
    You can use {prompting_category} as context to find the values./
    You need to deliver a dictionary as an output format (key: value(s)).
    You must always follow the Rule and avoid what is stated in the Restriction.
    '''
    # Extracted fields in dict format extraction using GPT-3-turbo
    print("----- request -----")

    print("----- standard request -----")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            },
        ],
    )

    # printing completion
    print(completion.choices[0].message.content)

    # Pandas dataframe
    extracted_fields = completion.choices[0].message.content
    fields_df = pd.DataFrame(index=list(extracted_fields), columns=COLUMNS)

    return fields_df
--------
f"""
    Act as the best financial analyst in the world.
    Analyze the following text and extract the values in the specified format. 
    Required values and formats:
    - ISIN: a string of 12 characters.
    - Issuer: a string representing the name of the issuer.
    - Currency: the currency code.
    - Launch Date: date in mm/dd/yyyy format.
    - Final Val. Day: date in mm/dd/yyyy format.
    - Maturity Date: date in mm/dd/yyyy format.
    - Underlying: the name of the underlying asset.
    - Strike: percentage (not equal to 100%).
    - Cap: percentage (not equal to 100%).
    - Barrier: percentage (not equal to 100%).

    Text: {text}

    Based on the analysis, provide the extracted information in a structured format with the category and corresponding value.
    """
