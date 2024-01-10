import json
import os
import sys
from dotenv import load_dotenv
import openai
load_dotenv()
# Set your OpenAI API keys
openai.api_key = os.environ.get("OPENAI_SECRET")
print("salut")
def translate_file(file_name):
    # Read the contents of the file
    with open(file_name, 'r') as file:
        data = json.load(file)

    print("coucou")
    # Translate each row
    for item in data:
        english_word = item['english']
        # Make the translation API call
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=f'Translate the word "{english_word}" to Portuguese, French, Spanish, Italian and German, return a json format like that : {{"Portuguese": "word", "French": "word", "Spanish": "word", "Italian": "word", "German": "word"}}',
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )
        # print(response.choices[0].text)
        # Extract the translations from the API response
        print(response.choices[0].text)
        translations = json.loads(response.choices[0].text)
        # Update the item with the translations
        # print("translation 0 : ", translations["Portuguese"])
        item['portuguese'] = translations["Portuguese"]
        item['french'] = translations["French"]
        item['spanish'] = translations["Spanish"]
        item['italian'] = translations["Italian"]
        item['german'] = translations["German"]


    # Save the translated data back to the file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=2)

def main():
    file_name = sys.argv[1]
    translate_file(file_name=file_name)

if __name__ == "__main__":
    main()