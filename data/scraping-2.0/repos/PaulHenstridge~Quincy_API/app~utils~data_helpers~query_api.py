import os
import openai
import requests
import tiktoken
#tiktoken used to count tokems
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

#truncate text to max-token limit - adjust for larger model!
def truncate_text(text, max_tokens=2048):
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[ :max_tokens]
    truncated_text = enc.decode(truncated_tokens)
    return truncated_text

def query_API(article):
    trimmed_article = truncate_text(article)
    # openai.organization =  os.getenv("OPENAI_ORG_KEY")
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # model_list = openai.Model.list()

    # Endpoint URL
    url = "https://api.openai.com/v1/chat/completions"

    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}" 
    }

    # Request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", 
             "content": """Your task is to generate a list of no more than 10 searchable keywords that summarize an article related to coding,
             software development, and tech careers. Each keyword should be 1-2 words long and directly relevant to the article's main focus.
             The list should only contain keywords, separated by commas, with no additional text. Prioritize the following types of keywords:

            1. Author's Name: Include the authors name where possible.
            2. Technologies: Mention specific programming languages or technologies (e.g., Python, React, Rust, Git, Firebase).
            3. Concepts: Include coding principles or methodologies (e.g., TDD, SOLID, Agile, accessability).
            4. Activities: Highlight specific tasks or projects related to the article (e.g., web scraping, deploying apps, APIs, data analysis, machine learning).

            Avoid using vague or generic terms like "course," "teaching," or "learn." 
            after processing the text, select only the 10 most relevant, searchable tags that will be of use to a student searching for learning materials, and return only those.
            Your output should look like this:
            Javascript, Node.js, web scraping, MongoDB, React, John Doe, TDD, algorithms, API, unit testing.
"""},
            { "role": "user", "content": trimmed_article }
        ],
    }

    # Making the POST request
    response = requests.post(url, headers=headers, json=data).json()

    # Printing the response
    try:
        return response["choices"][0]["message"]["content"]
    except KeyError:
        print("Error: Unexpected response format:", response)
        return ""  #empty string returned if no response, empty tags list can be added, avoid Nonetype errors later in code

   




