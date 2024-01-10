import openai
import os
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")

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

def main():
    parser = argparse.ArgumentParser(description="Multilingual Translation CLI")
    parser.add_argument("text", help="The text you want to translate")
    parser.add_argument("source_language", help="The source language of the text")
    parser.add_argument("target_language", help="The target language to translate the text into")

    args = parser.parse_args()

    translated_text = translate_text(args.text, args.source_language, args.target_language)
    print(f"Translated Text: {translated_text}")

if __name__ == "__main__":
    main()

# python CLI_translator.py "It's good to see you again." English Korean
