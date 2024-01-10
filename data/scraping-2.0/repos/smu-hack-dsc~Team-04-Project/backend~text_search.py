import openai
import re
import os
from dotenv import load_dotenv
import json
load_dotenv()

def getCategories(openaiApi,input):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # # for keywords
    # prompt = "Given the prompt \"" + input + "\" , separate each word in the input and please give me the singular and plural form of each word all in a single array. If a colour is in the input, please put the colour in another array. Return me one array with singular/plural array as the first element, and the colour array as the second element. Help me to format the array such that python functions can accept it. Only include words that are in the input. Respond with the array without any other words" 
    # completion = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     temperature = 0.2,
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # print(completion.choices[0].message.content.split("\n"))
    # print("+==========")
    
    # for categories
    prompt = "Given the prompt \"" + input + "\" please give me the color, clothing type and brands that are in the prompt. Only give the categories if the word is fully in the prompt. There can be more than one of each category. Please respond only with the categories without any other words and capitalise the first letter of the category. If any categories are empty, respond with N/A. For clothing type, please response with categories from this list [Tops, Bottoms, Dresses, Jumpsuits & Rompers, Outerwear, Suits, Accessories, Maternity, Jackets & Vests]" 
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature = 0.2,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    unprocessedArr = completion.choices[0].message.content.split("\n")
    finalArr = []
    regex_pattern = r".*?: "
    for categories in unprocessedArr:
        if categories != "":
            finalArr.append(re.sub(regex_pattern, "", categories).split(", "))
            
    json_data = json.dumps(finalArr)
    print(finalArr)
    return json_data