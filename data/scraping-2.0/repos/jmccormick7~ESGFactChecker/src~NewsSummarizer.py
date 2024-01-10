import os
import openai
from .NewsScraper import *

openai.api_key = "" #ENTER KEY HERE (pull from local env variable)

#from a json representation of news articles, generate a string gpt input asking for a summary
def generate_gpt_input(article_json):
    merged_articles = ""
    for article in article_json['articles']:
        title = article['title']
        summary = article['summary']
        merged_articles = merged_articles + "\nTitle: " + title
        merged_articles = merged_articles + "\nSummary: " + summary + "\n"
    
    prompt_preamble = "Summarize key ESG metrics and statistics from the following articles in five bullet points:\n\n"

    prompt = prompt_preamble + merged_articles
    print(prompt)
    return(prompt)

def generate_gpt_output(prompt, output_name):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
            "role": "system",
            "content": prompt
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    output = response['choices'][0]['message']['content']

    #commit gpt response to memoization file:
    outdirectory_path = "./src/gpt_outputs"
    if not os.path.exists(outdirectory_path):
        os.makedirs(outdirectory_path)
    outfile = output_name.lower() + ".txt" #squash names to lowercase for lookup
    outfile_path = os.path.join(outdirectory_path, outfile)
    with open(outfile_path, "w") as out:
            out.write(output)
    
    #return output
    return output

#retrieves gpt output from name. checks for memoized response first
def get_gpt(company_name):
    directory_path = "./src/gpt_outputs"

    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            if company_name.lower() in filename:
                print("cached")
                with open(os.path.join(directory_path, filename), "r") as file:
                    out = file.read()
                    return out

    print("fetched")       
    return generate_gpt_output(generate_gpt_input(get_news(company_name)), company_name)

firm_name = "Blackstone"
get_gpt(firm_name)