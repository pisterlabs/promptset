from flask import Flask, request
import openai

# Set the base URL for the OpenAI API
openai.api_base = "http://localhost:4891/v1"
# openai.api_base = "https://api.openai.com/v1"

openai.api_key = "not needed for a local LLM"

app = Flask(__name__)

# Global variable to store the initial prompt
prompt = "Hi"

# Main route to display the current prompt
@app.route('/')
def index():
    return prompt

# Route to receive the value via HTTP and update the prompt
@app.route('/update-prompt', methods=['POST'])
def update_prompt():
    global prompt
    prompt = request.form['value']
    return 'Prompt updated: ' + prompt

# Route to generate the response using the updated prompt
@app.route('/chat', methods=['POST'])
def generate_response():
    global prompt
 
    prompt = request.form['value']
    model = "vicuna-7b-1.1-q4_2"

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=50,
        temperature=.28,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False
    )
    return str(response)

if __name__ == '__main__':
    app.run()
