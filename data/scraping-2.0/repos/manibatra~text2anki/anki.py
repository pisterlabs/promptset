import openai
import json
import urllib.request
import sys

# Set up the OpenAI API client
openai.api_key = "<YOUR_API_KEY>"

ANKI_CONNECT_PORT = "8765"
MODEL = "gpt-4"

if len(sys.argv) < 2:
    print("Please provide the path to the text file as an argument.")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, 'r') as file:
    text = file.read()

def request(action, **params):
    return {'action': action, 'params': params, 'version': 6}


def invoke(action, **params):
    request_json = json.dumps(request(action, **params)).encode('utf-8')
    response = json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:' + ANKI_CONNECT_PORT, request_json)))
    if len(response) != 2:
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        raise Exception('response is missing required error field')
    if 'result' not in response:
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        raise Exception(response['error'])
    return response['result']


def generate_flashcards(text):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an AnkiAssistant that will create flashcards to be used in the "
                                          "Anki App."},
            {"role": "user", "content": f"You are an AnkiAssistant that will create flashcards to be used in the "
                                        f"Anki App. You should use HTML to format parts of the output according to "
                                        f"Anki format. Provide code examples and anything that assists in recall. "
                                        f"Separate the 'Front' and 'Back' of each flashcard with ||. Only use it once "
                                        f"in the flashcard."
                                        f"Every flashcard should be separated by '=======/"
                                        f"Create Anki flashcards from the following text: {text}."}

        ],
        temperature=0.2,
        presence_penalty=-0.2,
    )

    print(response)

    flashcards = response.choices[0].message['content'].strip().split('=======')
    return flashcards


def add_flashcards_to_anki(flashcards, deck_name):
    # Create the deck in Anki
    invoke('createDeck', deck=deck_name)

    # Add flashcards to the deck
    for flashcard in flashcards:
        if "||" in flashcard:
            front, back = flashcard.split("||")
            note = {
                "deckName": deck_name,
                "modelName": "Basic",
                "fields": {
                    "Front": front.strip().replace('Front:', ''),
                    "Back": back.strip().replace('Back:', '')
                },
                "tags": []
            }
            invoke('addNote', note=note)
        else:
            print(f"Skipping invalid flashcard: {flashcard}")


deck_name = "Generated Flashcards"
flashcards = generate_flashcards(text)

try:
    add_flashcards_to_anki(flashcards, deck_name)
except Exception as e:
    print(f"Error adding flashcards to Anki: {e}")
