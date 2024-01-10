import openai
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure your API key
openai.api_key = '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'

app = Flask(__name__)
CORS(app, methods=["GET", "POST", "OPTIONS"])  # This will allow all origins. For production, you should specify the allowed origins.

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data['question']

    # Process the question with GPT and get the response
    print('received a request')
    response = ask_huberman(question)

    return jsonify({"response": response})

def ask_huberman(question):
    # Instruction and context formatting
    instruction = (
        "You are Andrew Huberman, and specifically discuss scientific data "
        "that has been discussed on Andrew Huberman's podcast episodes. "
        "You should not mention Andrew Huberman in third person, as that is who you are. "
        "Your job is to answer questions that relate to the topics discussed in Andrew Huberman's podcasts. "
        "Your job is to offer a high level summary, and practical tips on how to incorporate the advice into the user's life. "
        "You are helping people improve people's lives."
    )

    # Merging the instruction, context, and question
    prompt = f"{instruction}\n\n{question}"

    try:
        # Making the API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{instruction}\n"},
                {"role": "user", "content": question}])

        # Extracting and returning the response
        print(response)
        # answer = response['choices'][0]['content'].strip()
        answer = response['choices'][0]['message']['content']
        return answer

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    app.run(port=4000)
