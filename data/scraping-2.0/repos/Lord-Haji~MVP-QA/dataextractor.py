import os
import openai

from config import API_KEY

openai.api_key = API_KEY



def extract_data(transcript):


    extract_prompt = """
    Current customer: Yes/No/Unsure
    Life Support: YES/NO


    Name : 
    DOB: 
    Contact Number : 
    Email: 
    Postal Address: 
    Solar : 
    Delivery method : 

    Did Agent confirm OTP (current customer only):

    Concession Card: 
    Issue Date : 
    Expiry Date : 

    Secondary Account Holder Name : 
    DOB : 

    Contact Number : 

    Billing cycle offered : 

    Promotions mentioned : 

    Supply Address: 
    NMI : 
    MIRN : 

    NOTE: If not confirmed then write - "NOT CONFIRMED"

        If agent did not capture from the customer's end then write - "NOT CAPTURED"


    Business name / Account name :
    ABN : 


    MOVE IN FEE: 
    Date of move in/Connection : 
    Power is currently off/ On : 
    Advised clear access for connection : Yes or No 
    """


    messages = [
        {
            "role": "system",
            "content": f"Extract and Identify all data mentioned in this transcript: '{transcript}'"
        },
        {"role": "user", "content": extract_prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
        max_tokens=340,
    )

    return response["choices"][0]["message"]["content"].strip()

# Example usage
# file_path = 'transcripts/movein_2.txt'
# file = open(file_path, 'r')

# transcript = file.read()

# print(classify_call(transcript))




