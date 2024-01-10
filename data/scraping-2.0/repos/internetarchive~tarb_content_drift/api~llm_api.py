import openai
import requests
from bs4 import BeautifulSoup
import streamlit as st
from flask import Flask, jsonify, request

# Set up OpenAI API credentials
openai.api_key = "openapi-key"

app = Flask(__name__)


# Define the GPT-3.5 prompt
prompt = """
Given a URL and a context string, check the relevance of the URL to the context string, and give approximate values for the below mentioned fields.
URL is the link of the url, Description is the brief description of the contents of the URL. Context is the given context text and Description is a brief description of the context text. 
Relevancy Percentage is a measure of how relevant the contents of the page are to the given context text.
Relevancy Percentage is always strictly a number, without any extra text, and should only be > 95% if the context string is absolutely the same as the given url content.

URL: {url}
Description: {url_description}

Context: {context}
Description: {context_description}

Relevancy Percentage: """

reason_prompt = """
The content in the URL: {url} (description: {url_description}) 
and the context: {context} (description: {context_description}) 
have a relevance of {relevancy_percentage}. 

Explain: """


def extract_url_content(url):
    """
    Extracts contents of a given URL.

    Args:
    url (str): webpage URL

    Returns:
    content (str): text content of the webpage
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # remove script and style contents
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text()
    content = " ".join(text.split())

    return content


# Function to generate descriptions using GPT-3.5
def generate_description(input_text):
    prompt = f"Describe the contents of the input:\n{input_text}\nDescription:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    description = response.choices[0].text.strip()
    return description


def generate_reason(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=input_text,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    reason = response.choices[0].text.strip()
    return reason


# Function to check relevancy using GPT-3.5
def check_relevancy(url, context):
    # Extract URL content
    url_content = extract_url_content(url)

    # Check if content exceeds 4096 tokens
    if len(url_content) > 4096:
        # Truncate URL content to fit within the limit
        url_content = url_content[:4096]

    # Describe the URL content and context
    url_description = generate_description(url_content)
    context_description = generate_description(context)

    # Format the prompt with the URL and context
    formatted_prompt = prompt.format(
        url=url,
        url_description=url_description,
        context=context,
        context_description=context_description,
    )

    # Generate response using GPT-3.5
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=formatted_prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Extract the relevancy percentage from the response
    relevancy_percentage = response.choices[0].text.strip()

    # Generate the reason for the relevancy score
    reason = generate_reason(formatted_prompt)

    return {
        "relevancy_percentage": relevancy_percentage,
        "url_description": url_description,
        "context_description": context_description,
        "reason": reason,
    }


app = Flask(__name__)


@app.route("/llm_relevance", methods=["POST"])
def calculate_llm_relevance():
    data = request.get_json()
    url = data.get("webpage")
    context_string = data.get("context_string")

    if url and context_string:
        try:
            result = check_relevancy(url, context_string)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Please supply webpage and context_string"}), 400


if __name__ == "__main__":
    app.run(port=5002)
