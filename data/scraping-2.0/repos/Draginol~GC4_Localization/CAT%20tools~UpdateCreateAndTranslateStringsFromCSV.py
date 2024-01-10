import csv
import os
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog
import openai
import time
import xml.dom.minidom

additional_strings_version = "v23"  # Variable to set the version of additional strings file

def prettify_xml(element):
    """
    Return a pretty-printed XML string for the Element.
    Ensures text nodes stay on the same line and includes XML declaration.
    """
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    
    # This function walks the DOM and strips CR/LF from text nodes
    def walk_node(node):
        if node.nodeType == xml.dom.Node.TEXT_NODE:
            node.data = node.data.replace('\n', '').replace('\r', '')
        for child in node.childNodes:
            walk_node(child)

    walk_node(reparsed)

    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Ensure xml declaration is included
    if not pretty_xml.startswith('<?xml'):
        pretty_xml = '<?xml version="1.0" encoding="utf-8"?>\n' + pretty_xml

    return pretty_xml

def write_pretty_xml(tree, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(prettify_xml(tree.getroot()))

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
    

def create_string_table(label, string):
    string_table = ET.Element("StringTable")
    label_element = ET.SubElement(string_table, "Label")
    label_element.text = label
    string_element = ET.SubElement(string_table, "String")
    string_element.text = string
    return string_table

def update_or_create_string(label_to_find, original_text, language, xml_directory):
    found = False
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
                    found = True
                    break
            if found:
                break

    if not found:
        if not found:
            additional_file_path = os.path.join(xml_directory, f"AdditionalStrings_{additional_strings_version}.xml")
            if os.path.exists(additional_file_path):
                additional_tree = ET.parse(additional_file_path)
                additional_root = additional_tree.getroot()
            else:
                additional_root = create_string_table_list()  # Updated line
                additional_tree = ET.ElementTree(additional_root)

        new_string_table = create_string_table(label_to_find, translate_text(label_to_find, original_text, language))
        additional_root.append(new_string_table)
        additional_tree.write(additional_file_path, encoding='utf-8', xml_declaration=True)
        print(f"Added new string {label_to_find} in {language}/AdditionalStrings_{additional_strings_version}.xml")

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

                found = False
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
                                found = True
                                break
                        if found:
                            break
                
                # Handling the case where the string is not found in existing files
                if not found:
                    additional_file_path = os.path.join(xml_directory, f"AdditionalStrings_{additional_strings_version}.xml")
                    if os.path.exists(additional_file_path):
                        additional_tree = ET.parse(additional_file_path)
                        additional_root = additional_tree.getroot()
                    else:
                        additional_root = ET.Element("StringTableList")
                        additional_tree = ET.ElementTree(additional_root)

                    new_string_table = create_string_table(label_to_find, translate_text(label_to_find, original_text, language))
                    additional_root.append(new_string_table)
                    write_pretty_xml(additional_tree, additional_file_path)   
                    print(f"Added new string {label_to_find} in {language}/AdditionalStrings_{additional_strings_version}.xml")



def main():
    csv_file = select_file()
    if not csv_file:
        print("No file selected.")
        return

    root_dir = os.path.dirname(csv_file)
    update_xml_files(csv_file, root_dir)

if __name__ == "__main__":
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if openai_api_key is None:
        print("OPENAI_API_KEY is not set.")
    else:
        print("OpenAI API Key found.")
        main()

