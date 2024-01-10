import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
import openai
import os

openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    print("OPENAI_API_KEY is not set.")
else:
    print("OpenAI API Key found.")
    # You can now use openai_api_key in y

class XLIFFTranslator:
    def __init__(self):
        self.root_dir = self.get_root_directory()
        self.process_files()

    def get_root_directory(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        dir_name = filedialog.askdirectory(title='Select Root Directory')
        return dir_name

    def process_files(self):
        for subdir, _, files in os.walk(self.root_dir):
            if os.path.basename(subdir) == self.root_dir:
                continue  # Skip the root directory itself
            target_language = os.path.basename(subdir)
            for file in files:
                if file.startswith("FlavorText_") and file.endswith(".xliff"):
                    self.process_xliff_file(os.path.join(subdir, file), target_language)

    def process_xliff_file(self, file_path, target_language):
        print(f"Processing file: {file_path} into {target_language} ")  # Output the file being processed
        tree = ET.parse(file_path)
        root = tree.getroot()
        for file in root.findall('.//file'):
            for trans_unit in file.findall('.//trans-unit'):
                target = trans_unit.find('target')
                # Check if target.text is None before calling strip()
                if target is not None and (target.text is None or target.text.strip() == ''):
                    source_text = trans_unit.find('source').text
                    translated_text = self.translate_to_language(source_text, target_language)
                    if translated_text:
                        target.text = translated_text
                        print(".")
                        tree.write(file_path, encoding='utf-8', xml_declaration=True)



    def translate_to_language(self, text, target_language):
        prompt = f"We are translating flavor text in a Sci-Fi video game we have to translate files into {target_language} while maintaining all formatting and maintainign the original meaning. Try to avoid long words or phrases or use spaces to enusre word wrap isn't broken. Return only the translated text. The text to translate is: {text}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in getting translation feedback: {e}")
            return None

if __name__ == "__main__":
    translator = XLIFFTranslator()
