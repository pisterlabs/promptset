import openai
import os
import pyperclip
from dotenv import load_dotenv, find_dotenv
import glob

# Load API Key
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

def read_files_from_directory(directory):
    """ Reads all files in a directory and returns their contents. """
    contents = ""
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            contents += file.read() + "\n\n"
    return contents

def get_response(prompt):
    """ Sends a prompt to the OpenAI API and returns the response. """
    instruction = (
    "Du bist ein spezialisierter Übersetzer für wichtige rechtliche Dokumente in internationalen Rechtshilfeverfahren, "
    "und bist speziell gut darin, einmal getroffene Übersetzungsentscheidungen über verschiedene Übersetzungsanfragen konsistent einzuhalten. "
    "Gib direkt den in die Zielsprache übersetzen Text aus. Bitte gib Kommentare insbesondere dann aus, wenn es widersprüchliche Richtlinien "
    "in den Beispielen oder im Vokabular gibt, oder wenn Unsicherheiten in der Übersetzung bestehen. Verwende die Zeichenfolge 'KKKKKK' "
    "um Kommentare vom übersetzten Text zu trennen."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return response.choices[0].message['content']


def read_all_example_contents():
    content = ""
    # Suche alle Ordner, die mit "example" beginnen
    for folder in glob.glob("example*/"):
        content += read_files_from_directory(folder) + "\n\n"
    return content

def main():
    # Read example texts and vocabulary
    examples_content = read_all_example_contents()
    vocabulary_content = read_files_from_directory("vocabulary")
    
    # Read new text to be translated
    with open("new text/german.txt", 'r', encoding='utf-8') as file:
        new_text = file.read()

    # Combine everything into one prompt with clear instructions in German
    prompt = (
    "Folgend sind Beispiele von zuvor übersetzten rechtlichen Dokumenten vom Deutschen ins Spanische. "
    "Halten Sie die Konsistenz mit diesen Übersetzungen bei ähnlichen Phrasen oder Begriffen aufrecht, und nehmen sie diese Beispiele als Orientierung für Stil und Wortwahl:\n\n"
    + examples_content
    + "\n\nUnten ist eine Liste von spezifischen juristischen Begriffen und deren festgelegten Übersetzungen. "
    "Verwenden Sie diese Begriffe konsistent in der Übersetzung:\n\n"
    + vocabulary_content
    + "\n\nÜbersetzen Sie den folgenden neuen juristischen Text vom Deutschen ins Spanische, "
    "und achten Sie dabei auf die Konsistenz mit den obigen Beispielen und die Verwendung des angegebenen Vokabulars:\n\n"
    + new_text
    )




    # Get the response from OpenAI
    translation = get_response(prompt)

    # Save the response
    with open("translated_text.txt", 'w', encoding='utf-8') as file:
        file.write(translation)

    # Copy the response to the clipboard
    pyperclip.copy(translation)

if __name__ == "__main__":
    main()
