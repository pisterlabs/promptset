"""
This script is designed to automate the localization process of string tables in a Sci-Fi video game. It utilizes OpenAI's GPT-4 model for translating game texts into multiple languages. The script performs the following functions:

1. It begins by checking for an OpenAI API key in the environment variables, which is necessary for accessing OpenAI's translation services.

2. The 'select_file' function enables the user to choose a CSV file through a file dialog. This CSV file is expected to contain two columns: 'Label' and 'String', where 'Label' is a unique identifier for each string and 'String' is the text that needs to be translated.

3. The 'translate_text' function takes a label, text, and a target language as input and generates a prompt for the OpenAI API to translate the text into the specified language. It respects the context of a Sci-Fi video game and maintains formatting codes in the text.

4. The 'update_xml_files' function reads the selected CSV file and iterates through each row. For each label and string, it attempts to translate the text into several predefined languages (like Chinese, French, German, etc.). The script locates the corresponding XML files in a directory structure organized by language and updates the string tables with the translated text.

5. After selecting the CSV file, the script identifies the root directory and updates the XML files in all the specified language folders with the translated texts.

Note: The script includes a delay between translation requests to avoid overloading the OpenAI API and handles errors in the translation process.

Usage:
- Ensure that the OPENAI_API_KEY is set in your environment variables.
- Run the script, select the CSV file with the original strings, and the script will handle the translations and updates.
"""


import csv
import os
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog
import openai
import time

openai_api_key = os.getenv('OPENAI_API_KEY')

import os

openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    print("OPENAI_API_KEY is not set.")
else:
    print("OpenAI API Key found.")
    # You can now use openai_api_key in your OpenAI API calls


def select_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

def translate_text(label_name, text, target_language):
    prompt = f"In the context of a Sci-Fi video game, given the string table entry label '{label_name}' as context, translate the following text into {target_language}. Respect all formatting codes and do not include the label. Add spaces without breaking meaning if a phrase is long to ensure word wrapping is not broken. Text to translate: {text}"

    try:
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in getting translation feedback: {e}")
        return None

def update_xml_files(csv_file_path, root_directory):
    languages = ["Chinese", "French", "German", "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Spanish"]

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label_to_find = row['Label']
            original_text = row['String']

            for language in languages:
                xml_directory = os.path.join(root_directory, language, 'Text')

                if not os.path.isdir(xml_directory):
                    print(f"Directory not found: {xml_directory}")
                    continue

                for xml_file in os.listdir(xml_directory):
                    if xml_file.endswith('.xml'):
                        xml_file_path = os.path.join(xml_directory, xml_file)
                        tree = ET.parse(xml_file_path)
                        root = tree.getroot()

                        for string_table in root.findall('.//StringTable'):
                            if string_table.find('Label').text == label_to_find:
                                translated_text = translate_text(label_to_find, original_text, language)
                                time.sleep(1)
                                if translated_text:
                                    string_table.find('String').text = translated_text
                                    tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
                                    print(f"Updated {label_to_find} in {language}/{xml_file}")
                                else:
                                    print(f"Failed to translate for {label_to_find} in {language}")
                                break

csv_file = select_file()
root_dir = os.path.dirname(csv_file)
update_xml_files(csv_file, root_dir)


