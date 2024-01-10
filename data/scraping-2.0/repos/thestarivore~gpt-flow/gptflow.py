# Import the os package
import os

import json

# Import the openai package
import openai

# Import a GPT Assistant
from gpt_agent import GPT_Agent
from gpt_agent import OpenAI_GPT_Agent
from web_search import web_search
from web_selenium import browse_website
from utils import count_string_tokens
  
def main():
    # Usage
    gptAgent = OpenAI_GPT_Agent()
    user_msg = "Find out and write on file the last 5 presidents of UnitedStates"

    response, json_content = gptAgent.make_first_decision(user_msg)
    gptAgent.print_response(response)

    if json_content["command"]["name"] == "google":
        print("\n---------------------------------------------\n")
        web_search_results = web_search(json_content["command"]["args"]["input"])
        print(web_search_results)
        web_search_results = json.loads(web_search_results)

        # Get the text for each website
        i = 0
        for item in web_search_results:
            text = browse_website(url=item["href"], question=user_msg, agent=gptAgent)
            web_search_results[i]["text"] = text
            print ("Nr. Tokens: " + str(count_string_tokens(string=text, model_name="gpt-3.5-turbo")))
            i = i + 1


if __name__ == "__main__":
    main()