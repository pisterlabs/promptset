import os
import xml.etree.ElementTree as ET
from tkinter import filedialog, Tk
from xml.dom.minidom import parseString
import openai
import time

class Translator:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key

    def translate_to_language(self, text, target_language):
        prompt = f"We are translating flavor text in a Sci-Fi video game we have to translate files into {target_language} while maintaining all formatting and maintaining the original meaning. Try to avoid long words or phrases or use spaces to ensure word wrap isn't broken. Return only the translated text. The text to translate is: {text}"
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

def get_all_xml_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xml')]

def get_internal_name_text_pairs(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if root.tag != "FlavorTextDefs":
        return []
    pairs = [(entry.find('InternalName').text, entry.find('FlavorTextOption/Text').text) for entry in root.findall('FlavorTextDef')]
    return pairs

def prettify_and_cleanup(xml_content):
    pretty_xml = parseString(xml_content).toprettyxml(indent="  ")
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    cleaned_content = '\n'.join(lines)
    return cleaned_content

def get_existing_translations(xliff_file):
    existing_translations = set()
    tree = ET.parse(xliff_file)
    root = tree.getroot()
    for file in root.findall('./file'):
        for unit in file.findall('./body/trans-unit'):
            internal_name = unit.attrib.get('internalName')
            if internal_name:
                existing_translations.add(internal_name)
    return existing_translations

def create_translated_file(filename, pairs, translator, target_language, existing_translations):
    root = ET.Element('FlavorTextDefs')
    for internal_name, text in pairs:
        if internal_name not in existing_translations:
            translated_text = translator.translate_to_language(text, target_language)
            time.sleep(1)
            print(".") # Progress indicator
            if translated_text:
                flavor_text_def = ET.SubElement(root, 'FlavorTextDef')
                ET.SubElement(flavor_text_def, 'InternalName').text = internal_name
                flavor_text_option = ET.SubElement(flavor_text_def, 'FlavorTextOption')
                ET.SubElement(flavor_text_option, 'Text').text = translated_text

    cleaned_xml = prettify_and_cleanup(ET.tostring(root, encoding="unicode", method="xml"))
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_xml)

def main():
    translator = Translator('your-openai-api-key')  # Replace with your actual OpenAI API key

    root = Tk()
    root.withdraw()
    english_directory = filedialog.askdirectory(title="Select the directory containing English XML FlavorText files")
    xml_files = get_all_xml_files(english_directory)

    file_to_pairs_map = {}
    for xml_file in xml_files:
        pairs = get_internal_name_text_pairs(xml_file)
        if pairs:
            file_to_pairs_map[os.path.basename(xml_file)] = pairs

    parent_directory = os.path.dirname(os.path.dirname(english_directory))
    language_dirs = ["Polish", "Chinese", "French", "German", "Greek", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish"]

    for lang_dir in language_dirs:
        full_lang_dir = os.path.join(parent_directory, lang_dir)
        if not os.path.exists(full_lang_dir):
            os.makedirs(full_lang_dir)
        xliff_files = [f for f in os.listdir(full_lang_dir) if f.endswith('.xliff')]
        for xliff_file in xliff_files:
            full_xliff_path = os.path.join(full_lang_dir, xliff_file)
            existing_translations = get_existing_translations(full_xliff_path)
            for filename, pairs in file_to_pairs_map.items():
                if xliff_file.endswith(filename + '.xliff'):
                    translated_filename = os.path.join(full_lang_dir, filename)
                    create_translated_file(translated_filename, pairs, translator, lang_dir, existing_translations)

if __name__ == "__main__":
    main()
