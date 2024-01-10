import openai
import os
import json


class GPTTranlatorInstation:
    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def gpt_internal_functions(self):
        functions = [
            {
                "name": "validate_translation",
                "description":  "Funkcja oceni czy dane translacje spełniają jej oczekiwania",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_pl": {
                            "type": "string",
                            "description": "Tłumaczenie wielkiego tłumacza język na polski",
                        },
                        "tekst_eng": {
                            "type": "string",
                            "description": "Tłumaczenie wielkiego tłumacza język na angielski",
                        }
                    },
                    "required": ["text_pl", "tekst_eng"],
                },
            }
        ]
        return functions
    
    def gpt_run_translation(self, input_text):
        openai.api_key = self.api_key

        #input_text= read_cli_input()

        messages = [
        {"role": "system", "content": "Jesteś wybinym poliglotą, który włada językami niczym czarodziej żywiołami, twoje translacje zotaną poddane obiektywnej ocenie, postaraj się tłumaczyć tak aby sens tekstu został w pełni zachowany uwazględzniając ukryte konteksty. Ludzie zwą cię wielkim tłumaczem"},
        {"role": "user", "content": f"Tekst do przetłumaczenia: '{input_text}', przekaż tłumaczenia" }]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=self.gpt_internal_functions(),
            function_call={"name": "validate_translation"}, 
        )
        response_message = response["choices"][0]["message"]

        #print(f"+++++++++{response_message}")
        
        clue_of_response = response_message["function_call"]["arguments"]
        parsed_clue = json.loads(clue_of_response)


        print(f"++<TO_ZWRACAM>++++{parsed_clue}")

        return parsed_clue

    