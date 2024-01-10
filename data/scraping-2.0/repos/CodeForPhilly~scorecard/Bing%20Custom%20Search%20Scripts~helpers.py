import requests 
import json
import time
#import openai
from bs4 import BeautifulSoup



def make_request(search_string,prefix,credentials):

    time.sleep(1.5)

    url = "https://api.bing.microsoft.com/v7.0/custom/search"
    headers = {
        "Ocp-Apim-Subscription-Key": f"{credentials}"
    }

    # Define the query parameters
    params = {
        "q": f'"{search_string}" + {prefix}',
        "customconfig": "2e4bf536-c98e-4f72-96b7-4a900c2e38ca",
        "mkt": "en-US",
        "answerCount":1
    }

    # Make the GET request
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:

        # # Save the response to a JSON file
        # with open("response_temp.json", "w") as json_file:
        #     json.dump(response.json(), json_file)
        
        return response.json()
    else:
        print(f"Request failed with status code {response.status_code}")
        return {}
    

def GPT_Summary(text):

    openai.api_key_path = "./GPT_ENV/.env"

    # #Get blurb using ChatGPT API
    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt=f"Summarize {text} in 50 characters or less",
    #     temperature=0.4,
    #     max_tokens=50000,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0
    # )

    # blurb = response["choices"][0]["text"].strip()
    
    # time.sleep(2)

    return "blurb"


def get_webpage_text(URL):

    time.sleep(1)

    response = requests.get(URL)

    if response.status_code == 200:
        
        soup = BeautifulSoup(response.text,'html.parser')

        #TODO:Properly pre-process this text to optimize likliehood of good summary
        webpage_text = soup.get_text()
        return webpage_text.replace("\n","")
    
    else:
        return "Error"

