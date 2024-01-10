import openai
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import asyncio
import random

load_dotenv()
gpt_key = os.getenv("GPT_KEY")
# system_message = "The output should be in the following format: 'Input: string, Output: { filters: { name: , no_of_ratings: , discount_price: }, intent: , short_description: , recommendation: }'. 'no_of_ratings' should be an integer between 0 and 4, 'discount_price' should be an integer, and 'short_description' should be a short description of the input string that can be searched on YouTube."
system_message = """You are a helpful assistant that reformulates user queries into more specific search queries. 

The output should be in the following format:

Name: 
No_of_Ratings:
Discount_price:
Intent:
Short_description:
Recommendation:


'no_of_ratings' should be an integer between 1 and 4 and if there is no rating in the query then make it 1, 'discount_price' should be an integer, and 'short_description' should be a short description of the input string that can be searched on YouTube. 
The recommendation should provide additional information or suggest a specific product that fits the user's criteria. "

"""


def gpt_response(query: str):

    # Define a prompt for GPT-3, including the user input and the system message
    prompt = f"{system_message} Translate the user input into a search query: {query}"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        api_key=gpt_key,
        temperature=0.2
    )


    # Extract the generated search query from the response
    output_string = response.choices[0].text.strip()

    # Print the generated search query for demonstration
    # print(output_string)

    # Split the output into lines
    output_lines = output_string.strip().split('\n')

    # Initialize variables to store the extracted values
    name = None
    no_of_ratings = None
    discount_price = None
    intent = None
    short_description = None
    recommendation = None

    output_list = []

    for line in output_lines:
        key, value = map(str.strip, line.split(':', 1))
        if key == 'Name':
            name = value
        elif key == 'No_of_Ratings':
            no_of_ratings = int(value)
        elif key == 'Discount_price':
            # Remove commas from the value
            value = value.replace(',', '')
            # Check if the price is given in thousands using 'k'
            if 'k' in value:
                # Remove the 'k' and multiply by 1000
                discount_price = int(value.replace('k', '')) * 1000
            else:
                discount_price = int(value)
        elif key == 'Intent':
            intent = value
        elif key == 'Short_description':
            short_description = value
        elif key == 'Recommendation':
            recommendation = value

   
    res = {
        "name": name,
        "ratings": no_of_ratings,
        "discount_price": discount_price,
        "recommendation": recommendation,
        "short_description": short_description

    }
    print(res)
    return res
