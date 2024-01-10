import openai
import os
from dotenv import load_dotenv, find_dotenv
import pyperclip

# Load API Key
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

def read_reconstruction_data():
    """Read additional reconstruction data from the file."""
    try:
        with open("reconstruction/reconstruction_data.txt", 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("Reconstruction data file not found. Continuing without additional data.")
        return ""

def reconstruct_text(original_text, additional_data):
    """ Send the text to OpenAI API for reconstruction using the chat model. """
    prompt = [
        {"role": "system", "content": "Ich bin an einer wichtigen Übersetzung von Dokumenten der spanischen Justiz. Leider hat die Texterkennung beim Scannen viele Fehler gemacht. Der Text muss so originalgetreu wie möglich rekonstruiert werden, um ihn danach automatisch übersetzen zu können. Das Textstück, dass ich dir hier zur Rekonstruktion übergebe, ist ein Stück eines grösseren Dokuments. Bitte rekonstruiere den spanischen Text so präzise und so wahrscheinlich wie möglich, und gib ihn dann in der Originalsprache aus. Als zusätzliche Kontextinformation übergebe ich dir Textstücke, die ich bereits aus anderen Teilen desselben Dokuments rekonstruiert habe, und auch in diesem zu rekonstruierenden Textstück vorkommen können. Es ist jedoch extrem wichtig, dass deine Ausgabe präzise das jetzt zu rekonstruierende Textstück so wahrscheinlich wie möglich rekonstruiert, und zwar jedes Zeichen so genau wie möglich, auch wenn in den Kontextinformationen, also in anderen Teilen des Dokuments als dem jetzt zu rekonstruierenden Textstück allenfalls davon abweichende Schreibweisen benutzt wurden:"},
        {"role": "system", "content": additional_data},
        {"role": "user", "content": original_text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=prompt,
        temperature=0.7
    )

    return response.choices[0].message['content'].strip()

def main():
    # Read the new text directly from 'new text/german.txt'
    try:
        with open("new text/german.txt", 'r', encoding='utf-8') as file:
            original_text = file.read()
    except FileNotFoundError:
        print("The file 'new text/german.txt' was not found.")
        return

    # Read additional reconstruction data
    additional_data = read_reconstruction_data()

    # Reconstruct the text
    reconstructed_text = reconstruct_text(original_text, additional_data)

    # Save the reconstructed text back into the same file and copy to clipboard
    with open("new text/german.txt", 'w', encoding='utf-8') as file:
        file.write(reconstructed_text)
    pyperclip.copy(reconstructed_text)

    print("The reconstructed text has been saved and copied to the clipboard.")

if __name__ == "__main__":
    main()


