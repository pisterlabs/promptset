import json
import os

import openai
from dotenv import load_dotenv
from global_variables import source_language, target_language
from models.note import Note

GPT_MODEL = "gpt-3.5-turbo-1106"
# gpt-3.5-turbo-1106 (cheaper) or gpt-4-1106-preview (more expensive)


class OpenAiInterface:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def call_api(self, systemPrompt: str, string: str):
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": systemPrompt,
                },
                {
                    "role": "user",
                    "content": string,
                },
            ],
        )

        if response.choices[0].finish_reason != "length":
            return response.choices[0].message.content

        raise Exception("Response terminated due to length.")

    def look_up_tokens(self, text):
        system_prompt = f"""
            The task is to parse a given string in {source_language.capitalize()} and return an 
            array of JSON objects. Each JSON object corresponds to a bracketed token in the input 
            string. Ensure these rules are followed:

            - The response must be an array containing JSON objects, even if there's only one 
              bracketed token.
            - The array length should match the number of bracketed tokens in the input string.
            - Return only the raw array without markdown formatting or other text decoration.

            Structure each JSON object in the array as follows:

            {{
                "token": "<The exact bracketed token or token fragment from the input>",
                "pos": "<Part of Speech tags like 'NOUN', 'VERB', etc.>",
                "source": "<the source word, usually the same as 'token'. verbs in the infinitive, 
                          otherwise do not change inflection from input>",
                "target": "<{target_language.capitalize()} translation, with verbs in the 
                          infinitive>",
                "gender": "<'MASC', 'FEM', or null when inapplicable. almost never null for nouns 
                          and adjectives>",
                "number": "<'SING', 'PLUR', or null when inapplicable. almost never null for nouns 
                          and adjectives>"
            }}

            - 'gender' and 'number' must not be null if the token's gender/number is discernible 
              independently.
            - For verbs, always use the infinitive form in 'source' and 'target' fields.
            - Exclude parts of a token not enclosed in brackets.

            Example input and output scenarios:

            Numerical Value:
            Input: "[onze]"
            Output: {{
                "token": "onze",
                "pos": "NUM",
                "source": "onze",
                "target": "eleven",
                "gender": null,
                "number": null
            }}

            Noun:
            Input: "[heures]"
            Output: {{
                "token": "heures",
                "pos": "NOUN",
                "source": "heures",
                "target": "hours",
                "gender": "FEM",
                "number": "PLUR"
            }}

            Verb (non-infinitive form):
            Input: "[regarda]"
            Output: {{
                "token": "regarda",
                "pos": "VERB",
                "source": "regarder",
                "target": "to look",
                "gender": null,
                "number": null
            }}

            Determiner:
            Input: "[mon]"
            Output: {{
                "token": "mon",
                "pos": "DET",
                "source": "mon",
                "target": "my",
                "gender": "MASC",
                "number": "SING"
            }}

            Contraction Fragment:
            Input: "[Ferme]-la"
            Output: {{
                "token": "Ferme",
                "pos": "VERB",
                "source": "fermer", // infinitive because VERB
                "target": "to close", // infinitive because VERB
                "gender": null,
                "number": null
            }}
        """

        request = text.get_marked_string()
        print(f"OpenAI request: {request}")

        raw_response = self.call_api(system_prompt, request).strip()
        # for debugging
        print(f"raw_response: {raw_response}")

        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]

        response = json.loads(raw_response)
        # for debugging
        # print(f"OpenAI response: {json.dumps(response, indent=4)}")

        marked_tokens = text.get_marked_tokens()

        if len(response) != len(marked_tokens):
            print(
                f"\n\033[31mWARN:\033[0m received different number of responses than requests sent"
            )

        for marked_token in marked_tokens:
            response_match = None

            for item in response:
                if item.get("token") == marked_token.text.string:
                    response_match = item
                    break

            # TODO: Ask again for items that were missed. Can pretty much just do it by running the
            # above again.
            if response_match is None:
                print(
                    f"\033[31mWARN:\033[0m skipping {marked_token.text.string} "
                    f"(did not find matching response item)"
                )
                continue

            pos = item.get("pos")
            source = item.get("source")
            target = item.get("target")
            gender = item.get("gender")
            number = item.get("number")

            note = Note(pos, source, target, None, gender, number)

            marked_token.add_note(note)
