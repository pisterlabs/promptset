from flask import Flask, request, jsonify, render_template
import os
import openai
import logging

logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__, static_folder='../frontend')
app = Flask(__name__, template_folder='../frontend')
openai_api_key = os.environ.get("OPENAI_API_KEY")


@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/test', methods=['GET'])
# def test():
#     return "Test endpoint"

# @app.route('/')
# def home():
#     return send_from_directory(app.static_folder, 'index.html')


def detect_language(text):
    # Count the number of non-ASCII characters
    non_ascii_count = sum(1 for char in text if ord(char) >= 128)

    # If more than half of the characters are non-ASCII, assume it's Mandarin Chinese
    if non_ascii_count > len(text) / 2:
        return "Mandarin Chinese"
    else:
        return "English"


def translate_text(text, source_lang, target_lang):
    print(f"Translating text: {text}, from {source_lang} to {target_lang}")  # Debugging line
    if target_lang == "Mandarin Chinese":
        target_lang = "Simplified Chinese"  # Specify Simplified Chinese
    prompt = f"Translate the following text from {source_lang} to {target_lang} (Simplified Chinese characters only): {text}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        api_key=openai_api_key  # Use the API key here
    )
    return response.choices[0].text.strip()


@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        text = data['sourceText']
        source_language = data['sourceLang']
        target_language = data['targetLang']

        # Convert the language names to the format expected by the translation function
        if source_language == "English":
            source_language = "English"
        else:
            source_language = "Mandarin Chinese"

        if target_language == "English":
            target_language = "English"
        else:
            target_language = "Mandarin Chinese"

        translated_text = translate_text(text, source_language, target_language)
        return jsonify(
            {"translatedText": translated_text, "sourceLanguage": source_language, "targetLanguage": target_language})
    except Exception as e:
        app.logger.error(f"Exception occurred: {str(e)}")
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=10000)
