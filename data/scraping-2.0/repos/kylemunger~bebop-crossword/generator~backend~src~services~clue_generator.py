from openai import OpenAI
import json

class ClueGenerator:
    def __init__(self, client:OpenAI):
        self.client = client
        self.model = "gpt-4-1106-preview"
        self.fnname = "generate_crossword_clues"

    def generate_clues(self, word):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.generate_clues_messages(word),
            tools=self.generate_clues_schema(),
            tool_choice={"type": "function", "function": {"name": self.fnname}}
        )

        j = json.loads(completion.model_dump_json())
        call = j['choices'][0]['message']['tool_calls'][0]['function']['arguments']
        return json.loads(call)['clues']

    def generate_clues_and_words(self):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.generate_clues_and_words_messages(),
            tools=self.generate_clues_and_words_schema(),
            tool_choice={"type": "function", "function": {"name": self.fnname}}
        )

        j = json.loads(completion.model_dump_json())
        call = j['choices'][0]['message']['tool_calls'][0]['function']['arguments']
        return json.loads(call)['clues']

    def generate_clues_messages(self, word):
        return [
            {"role": "system", "content": "You are a crossword generating bot. You will generate creative clues for a given input word. The user input will simply be the word to generate clues for, and you will generate 3 creative clues for that word. Each clue should simply be a string representing one clue, there should not be any length, double clues, or punctuation"},
            {"role": "user", "content": word}
        ]

    def generate_clues_and_words_messages(self):
        return [
            {"role": "system", "content": "You are a crossword generating bot. You will generate creative crossword words and clues for a given input word."},
            {"role": "user", "content": "Please generate 5 clues"}
        ]

    def generate_clues_schema(self):
        return [{
            "type": "function",
            "function": {
                "name": "generate_crossword_clues",
                "description": "Creates creative crossword clues based on an input word",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clues": {
                            "type": "array",
                            "description": "Crossword clues generated based on the input word",
                            'items': {
                                "type": "string",
                                "description": "A crossword clue"
                            }
                        }
                    },
                    "required": ["clues"],
                },
            }
        }]

    def generate_clues_and_words_schema(self):
        return [{
            "type": "function",
            "function": {
                "name": "generate_crossword_clues",
                "description": "Creates creative crossword clues",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clues": {
                            "type": "array",
                            "description": "Crossword clues generated",
                            'items': {
                                "type": "object",
                                "properties": {
                                    "clue": {
                                        "type": "string",
                                        "description": "A crossword clue"
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "The answer to the crossword clue"
                                    }
                                },
                                "required": ["clue", "answer"]
                            }
                        }
                    },
                    "required": ["clues"],
                },
            }
        }]
