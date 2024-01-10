from flask import Flask, request, jsonify
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Define the translation function
def translate_text(text, source_language, target_language):
    prompt = f"Translate the following '{source_language}' text to '{target_language}': {text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    translation = response.choices[0].message.content.strip()
    return translation

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()

    text = data.get("text")
    source_language = data.get("source_language")
    target_language = data.get("target_language")

    if not text or not source_language or not target_language:
        return jsonify({"error": "Missing required parameters"}), 400

    translated_text = translate_text(text, source_language, target_language)

    return jsonify({"translated_text": translated_text})

if __name__ == '__main__':
    app.run(debug=True)


# curl -X POST -H "Content-Type: application/json" -d '{"text": "Hello, world!", "source_language": "English", "target_language": "Korean"}' http://localhost:5000/translate