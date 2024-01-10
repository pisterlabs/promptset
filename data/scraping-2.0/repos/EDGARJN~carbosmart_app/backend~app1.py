import langchain
import openai
import re
from flask import Flask, request, jsonify

# Set up the Langchain client
# langchain_client = langchain.Client()

# Set up the OpenAI API key
openai.api_key = "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"

# Load the data about yourself from an external text file
with open("data.txt", "r") as f:
    data = f.read()

# Define a function to extract information about yourself from the data
def extract_info(query):
    # Use regular expressions to extract information from the data
    if re.search(f"\\b{query}\\b", data):
        return re.findall(f"\\b{query}\\b:\\s*(.*?)\\n", data)[0]
    else:
        return None


# Define a function to generate a response using ChatGPT.
def generate_response(prompt):
    # Use ChatGPT to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

# Set up the chatbot server
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    # Get the query from the JSON request
    request_data = request.get_json()
    query = request_data["query"]
    
    # Handle the query and return the response
    response = handle_query(query)
    return jsonify({"response": response})

def handle_query(query):
    # Extract information about yourself from the query
    info = extract_info(query)
    if info:
        return info
    else:
        # Generate a response using ChatGPT
        prompt = f"What can you tell me about {query}? but if asked about something relate to {data} give me your response in swahili language"
        response = generate_response(prompt)
        return response

if __name__ == "__main__":
    app.run(debug=True)
