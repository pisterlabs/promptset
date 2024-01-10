import json
import os
import openai

OUTPUT_FOLDER = 'out/'


class AICompiler:
    def __init__(self, path, open_ai_key):
        openai.api_key = open_ai_key
        self.path = path
        self.role_prompt = """You must only provide code as an answer. You dont write anything but code. You cant communicate with the user in any other way. and remove the comments from the code"""
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + OUTPUT_FOLDER):
            os.makedirs(path + OUTPUT_FOLDER)

    @staticmethod
    def _extract_code(response):
        code = ""
        code_started = False
        for line in response.splitlines():
            if "```" in line:
                if code_started:
                    return code
                else:
                    code_started = True
            elif code_started:
                code += line + "\n"

        if code == "":
            return response
        return code

    def compile_program(self, filename_in, filename_out):
        with open(self.path + filename_in, 'r') as file:
            ai_program = json.load(file)

        messages = []
        messages.append({'role': 'system', 'content': ai_program['role_prompt']})
        messages.extend(ai_program['messages'])

        response = openai.ChatCompletion.create(
            model=ai_program['model'],
            messages=messages,
            temperature=ai_program['temperature'],
            top_p=ai_program['top_p'],
            frequency_penalty=ai_program['frequency_penalty'],
            presence_penalty=ai_program['presence_penalty'],
        )

        with open(self.path + OUTPUT_FOLDER + filename_out, 'w') as outfile:
            code = AICompiler._extract_code(response['choices'][0]['message']['content'])
            outfile.write(code)

    def _create_program(self, messages):
        return {
            "languages": ["javascript", "python", "dart", "java", "c++", "c#"],
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "role_prompt": self.role_prompt,
            "messages": messages
        }

    def write_and_compile_program(self, name, messages, extension):
        program = self._create_program(messages)
        if os.path.exists(self.path + name + '.ai.json'):
            with open(self.path + name + '.ai.json', 'r') as file:
                # if program is equal to the one we want to write, do nothing compare by text
                if file.read() == json.dumps(program):
                    return False
        self.write_program(name, messages)
        self.compile_program(name + '.ai.json', name + '.ai.' + extension)

    def write_program(self, name, messages):
        obj = self._create_program(messages)
        # if program previously existed, read it


        with open(self.path + name + '.ai.json', 'w') as outfile:
            json.dump(obj, outfile)
