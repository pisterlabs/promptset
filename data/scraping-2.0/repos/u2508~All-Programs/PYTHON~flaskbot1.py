from flask import Flask, request
import openai
import json

# Set up the Flask app
app = Flask(__name__)

# Set up the OpenAI API credentials
openai.api_key = "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"

# Set up the GPT model parameters
model_engine = "davinci" # use the most powerful GPT model
temperature = 0.7 # controls the "creativity" of the AI's responses
max_tokens = 100 # maximum length of each response
stop_sequence = "\n" # end the response after the first line break

# Create the AI chatbot endpoint
@app.route("/chat", methods=["POST"])
def chat():
    # Get the user's message from the POST request
    user_message = request.form.get("message")

    # Use the OpenAI GPT model to generate a response
    response = openai.Completion.create(
        engine=model_engine,
        prompt=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequence
    )

    # Extract the response text from the OpenAI response object
    response_text = response.choices[0].text.strip()

    # Package the response as a JSON object and return it to the user
    return json.dumps({"response": response_text})

# Start the Flask app
if __name__ == "__main__":
    app.run()
