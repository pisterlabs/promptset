import openai
import os
import argparse
import json



class GPTInstance:
    def __init__(self, model, initial_messages, functions):
        self.model = model
        self.initial_messages = initial_messages
        self.functions = functions
        #self.function_call = function_call

def initialize_gpt():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    model="gpt-3.5-turbo"
    initial_messages=[
    {"role": "system", "content": "You are fluent native speaker in all of knowing languages, your task is to translate input text to diffrent language"},
    {"role": "user", "content": ""}
    ]

    '''
    initial_messages=[
    {"role": "system", "content": "You are fluent native speaker in all of knowing languages, your task is to translate input text to diffrent language"}
    ]'''

    '''initial_messages=[
    {"role": "system", "content": "You are fluent native speaker in all of knowing languages, your task is to translate input text to diffrent language"},
    {"role": "function", "name": translate_text, "content": ""}
    ]'''

    functions = [
    {
        "name": "translate_text",
        "description": "Get the translation of the input text to output language, function requires input text in Engish",
        "parameters": {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "Input text to translate",
                },
                "output_language": {
                    "type": "string",
                    "description": "The language to which the input text should be translated.",
                },
            },
            "required": ["input_text", "output_language"],
        },
    },
    ]

    #function_call={"name": "translate_text"},
    #function_call="auto",

    return GPTInstance(model, initial_messages, functions)

def translate_text(gpt_instance, text, output_language):
    
    function_call = {
        "name": "translate_text",
        "arguments": {
            "input_text": text,
            "output_language": output_language
        }
    }

    #extra_message = {"role": "function", "name": translate_text, "content": "Translate the following text to {output_language}: '{text}'"}
    #gpt_instance.initial_messages.append(extra_message)

    gpt_instance.initial_messages[1]['content'] = f"Translate the following text to {output_language}: '{text}'"
    response = openai.ChatCompletion.create(model=gpt_instance.model, messages=gpt_instance.initial_messages, functions=gpt_instance.functions, function_call=function_call )
    '''second_response = openai.ChatCompletion.create(model=gpt_instance.model, messages=[
        {"role": "system", "content": "You are fluent native speaker in all of knowing languages, your task is to translate input text to diffrent language"},
        {"role": "function", "name": translate_text, "content": f"Translate the following text to {output_language}: '{text}'"}
    ], functions=gpt_instance.functions, function_call=function_call )'''
    #translated_text = response.choices[0].message["content"]
    translated_text = response.choices[0]
    #translated_text = second_response.choices[0]


    return translated_text

def read_cli_input():
    parser = argparse.ArgumentParser(description="Text to translation")
    parser.add_argument("--text", type=str, help="Type text to translate")
    parser.add_argument("--language", type=str, help="Target language for translation")
    args = parser.parse_args()

    if args.text:
        text_from_cmd = args.text
    else: 
        text_from_cmd = '''Missing input CLI text'''

    language = args.language if args.language else "English" #Default language

    return text_from_cmd, language

def save_translation(input_text, translate_text):
    data = {
        "input_text": input_text, 
        "translation": translate_text
    }

    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_directory = os.path.abspath(os.path.join(script_directory, "..", ".."))
    file_path = os.path.join(project_directory, "fs", "translation.json")

    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)
        print(f"Translation saved: {file_path}")
    except Exception as e:
        print(f"Error during save to JSON: {e}")

    pass    

def main():
    """
    text_to_translate = '''Był sobie kiedyś prezes pewnej firmy zwiazanej z bezpieczeństwem przemysłowym.
      Charakter miał niezby dobry, ale znajomi nazywali go "byczkiem smolnym"'''
    """
    text_to_translate, target_language = read_cli_input()
    print(f"text_to_translate: {text_to_translate}")
    print(f"target_language: {target_language}")
    translate_instance = initialize_gpt()
    eng_txt = translate_text(translate_instance, text_to_translate, target_language)

    print(f"------------------")
    print(f"tekst po polsku: {text_to_translate}")
    print(f"------------------")
    print(f"tekst po angielsku: {eng_txt}")

    save_translation(text_to_translate,eng_txt)
    
    pass


main()
