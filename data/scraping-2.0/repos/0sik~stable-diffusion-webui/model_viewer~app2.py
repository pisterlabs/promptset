from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.json
        input_text = data['input_text']

        # Call the AI model to generate a response
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can choose an appropriate engine
            prompt=input_text,
            max_tokens=50  # You can adjust the length of the generated response
        )

        generated_text = response.choices[0].text.strip()

        return jsonify({"generated_text": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
