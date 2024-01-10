import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_info(filename):
    pre_prompt = ""
    with open(filename) as file:
        for lines in file:
            lines = lines.strip()
            pre_prompt += lines
    response = openai.Completion.create(
        model= "gpt-3.5-turbo-instruct",
        prompt="Can you descripe this program in detail and in proper documentation for developers reading the code\n " + pre_prompt ,
        temperature=.6,
        max_tokens=1000
    )
    return response

def ouput_file(filename,name):
    response = get_info(filename)
    text = response['choices'][0]['text']
    print(text)
    f = open(name+".txt", 'w')
    f.write(text)
    

    
    



def main():
    #filename = input("enter a file ")
    filename = "linear_search.py"
    name = "linear_search_documentation"
    ouput_file(filename,name)
main()

    

            

    