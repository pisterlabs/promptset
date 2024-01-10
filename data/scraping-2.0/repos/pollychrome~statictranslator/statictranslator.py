import argparse
import json
import os
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Language code mappings for easy_localization and OpenAI
LANGUAGE_MAPPINGS = {
    'en-us': ('en_US', 'English'),
    'pt-br': ('pt_BR', 'Brazilian Portuguese'),
    'es-es': ('es_ES', 'Spanish'),
    'hi-in': ('hi_IN', 'Hindi'),
    'de-de': ('de_DE', 'German'),
    'ar-sa': ('ar_SA', 'Arabic'),
    'fr-fr': ('fr_FR', 'French'),
    'tr-tr': ('tr_TR', 'Turkish'),
    'it-it': ('it_IT', 'Italian'),
    'fa-ir': ('fa_IR', 'Persian'),
    'pl-pl': ('pl_PL', 'Polish')
}

def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def estimate_token_count(text):
    return len(text.split()) + text.count(' ') + text.count('.') + text.count(',') + text.count('?') + text.count('!')

def estimate_token_count(text):
    return len(text.split()) + text.count(' ') + text.count('.') + text.count(',') + text.count('?') + text.count('!')

def translate_text_with_openai(text, target_language, openai_api_key, max_tokens=3000):
    openai.api_key = openai_api_key
    prompt = f"Translate the following English JSON content to {target_language}, maintaining the original key-value structure. Only translate the values, keeping the keys in English: {text}"

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"An error occurred during translation: {e}")
        return ""

def create_translation_batches(source_data, target_language, openai_api_key, max_batch_size=10, max_tokens=3000):
    translated_data = {}
    batch = []
    batch_keys = []

    for key, text in source_data.items():
        batch.append(text)
        batch_keys.append(key)

        if len(batch) >= max_batch_size:
            translated_batch = [translate_text_with_openai(text, target_language, openai_api_key, max_tokens) for text in batch]
            translated_data.update(dict(zip(batch_keys, translated_batch)))
            batch = []
            batch_keys = []

    if batch:
        translated_batch = [translate_text_with_openai(text, target_language, openai_api_key, max_tokens) for text in batch]
        translated_data.update(dict(zip(batch_keys, translated_batch)))

    return translated_data

def translate_and_save_json_with_directory_check(source_data, language_mappings, output_directory, openai_api_key):
    ensure_directory_exists(output_directory)

    for lang_code, (file_suffix, openai_language) in language_mappings.items():
        # Skip translation if the source and target languages are the same
        if openai_language.lower() == "english":
            logging.info("Skipping translation for English (source language).")
            continue

        logging.info(f"Starting translation for language: {openai_language}")
        translated_data = create_translation_batches(source_data, openai_language, openai_api_key)

        output_file = f"{output_directory}/{file_suffix}.json"
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(translated_data, file, ensure_ascii=False, indent=2)

        logging.info(f"Translated file saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Translate a JSON file to multiple languages using OpenAI.")
    parser.add_argument("source_file", type=str, help="Path to the source JSON file in English.")
    parser.add_argument("--output_dir", type=str, default="./translations", help="Output directory for translated files.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key. Must be provided as an argument.")

    args = parser.parse_args()

    with open(args.source_file, 'r', encoding='utf-8') as file:
        source_data = json.load(file)

    translate_and_save_json_with_directory_check(source_data, LANGUAGE_MAPPINGS, args.output_dir, args.api_key)

if __name__ == "__main__":
    main()
