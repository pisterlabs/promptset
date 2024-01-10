# from flask import Flask, request, jsonify, send_from_directory
# import os
# import openai

# app = Flask(__name__, static_folder='../frontend')
# openai.api_key = "UrGPTAPI"


# @app.route('/')
# def home():
#     return send_from_directory(app.static_folder, 'index.html')


# def detect_language(text):
#     # Count the number of non-ASCII characters
#     non_ascii_count = sum(1 for char in text if ord(char) >= 128)

#     # If more than half of the characters are non-ASCII, assume it's Mandarin Chinese
#     if non_ascii_count > len(text) / 2:
#         return "Mandarin Chinese"
#     else:
#         return "English"


# def translate_text(text, source_lang, target_lang):
#     print(f"Translating text: {text}, from {source_lang} to {target_lang}")  # Debugging line
#     prompt = f"Translate the following text from {source_lang} to {target_lang}: {text}"
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=100,
#     )
#     return response.choices[0].text.strip()


# @app.route('/translate', methods=['POST'])
# def translate():
#     data = request.json
#     text = data['sourceText']

#     detected_language = detect_language(text)

#     if detected_language == "English":
#         source_language = "English"
#         target_language = "Mandarin Chinese"
#     else:
#         source_language = "Mandarin Chinese"
#         target_language = "English"

#     try:
#         translated_text = translate_text(text, source_language, target_language)
#         return jsonify(
#             {"translatedText": translated_text, "sourceLanguage": source_language, "targetLanguage": target_language})
#     except Exception as e:
#         return jsonify({"error": str(e)})


# if __name__ == '__main__':
#     app.run(debug=True)
