from flask import Flask, render_template, request
import openai

app = Flask(__name__)

# Your OpenAI API key
api_key = "XXX"
openai.api_key = api_key

def generate_response(prompt):
    # Add the specific string to the beginning of the input
    prompt_with_intro = "Rate the SMS from 1 to 5 where 1 is SAFE and 5 is FRAUD. " + prompt

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_with_intro},
        ],
    )
    return response['choices'][0]['message']['content']

def classify_response(response_text):
    if any("rate" in response_text and num in response_text for num in ["4", "5"]):
        return "FRAUD"
    elif any("rate" in response_text and num in response_text for num in ["1", "2", "3"]):
        return "SAFE"
    else:
        return "Could not Identify"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    classification = None

    if request.method == 'POST':
        input_string = request.form['inputString']
        # Use GPT-3.5 Turbo to generate a response with the added intro string
        gpt_response = generate_response(input_string)
        classification = classify_response(gpt_response)
        #result = f"GPT-3.5 Turbo Response: {gpt_response}"

    return render_template('index.html', result=result, classification=classification)

if __name__ == '__main__':
    app.run(debug=True)
