import json
import openai

from django.conf import settings
from typing import Union

from ._diclookup import DictionaryLookUp


class OpenAI(DictionaryLookUp):
    def get_word_info(
        self,
        definition: str,
        en_translation: str,
        examples: list,
        is_informal: bool,
        is_mwe: bool,
        lemma: str,
        pos: str,
        source: str,
    ):
        word_info = {
            "pos": pos,
            "en_translation": en_translation,
            "is_informal": is_informal,
            "lemma": lemma,
            "definition": definition,
            "examples": examples,
            "is_mwe": is_mwe,
            "source": source,
        }
        return json.dumps(word_info)

    @staticmethod
    def look_up(
        lookup_word: str,
        lemma: str,
        lang_prefix: str,
        lang_source: str,
        lang_target: str,
    ) -> Union[list, None]:
        function_descriptions = [
            {
                "name": "get_word_info",
                "description": "Get word info.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "definition": {
                            "type": "string",
                            "description": "Definition of the word.",
                        },
                        "en_translation": {
                            "type": "string",
                            "description": "The translation into English.",
                        },
                        "examples": {
                            "type": "array",
                            "description": "Examples of the word in use.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "lang": {"type": "string"},
                                    "example": {"type": "string"},
                                },
                            },
                        },
                        "is_informal": {
                            "type": "boolean",
                            "description": "Whether or not the word is informal.",
                        },
                        "is_mwe": {
                            "type": "boolean",
                            "description": "Whether or not the word is a multi-word expression.",
                        },
                        "lemma": {
                            "type": "string",
                            "description": "Lemma",
                        },
                        "pos": {
                            "type": "string",
                            "description": "The part of speech.",
                        },
                        "source": {
                            "type": "string",
                            "description": "The URL source of the definition.",
                        },
                    },
                    "required": [
                        "definition",
                        "en_translation",
                        "examples",
                        "is_informal",
                        "is_mwe",
                        "lemma",
                        "pos",
                        "source",
                    ],
                },
            }
        ]

        user_prompt = f"""
        Lang source: {lang_source}
        Target language: {lang_target}
        Define the word {lookup_word} in {lang_target}.
        """

        openai.api_key = settings.OPENAI_API_KEY
        try:
            completion = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a multi-lingual dictionary.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                # Add function calling
                functions=function_descriptions,
                function_call="auto",
            )
        except openai.error.APIError as e:
            print(f"OpenAI error: {e}")
            return None

        output = completion.choices[0].message
        params = json.loads(output.function_call.arguments)
        params["llm"] = settings.OPENAI_MODEL
        return [params]
