import openai
import LLMConfigurator
import re
import json
import os
from io import StringIO
import sys
from typing import Optional, Dict

def run_python_code(command: str, _globals: Optional[Dict] = None, _locals: Optional[Dict] = None) -> str:
    _globals = _globals if _globals is not None else {}
    _locals = _locals if _locals is not None else {}

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    try:
        exec(command, _globals, _locals)
        sys.stdout = old_stdout
        output = mystdout.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        output = str(e)

    return output

class sapperchain:
    def __init__(self, OpenaiKey):
        openai.api_key = OpenaiKey
    def promptbase(self,prompt_template):
        self.prompt_template = json.loads(prompt_template)
    def worker(self, prompt, preunits, model):
        if(model["engine"].replace(" ","") == "PythonREPL"):
            return self.run_PythonREPL(prompt, preunits, model)
        else:
            return self.run_Function(prompt, preunits, model)
    def getPromptParams(self,prompt_template):
        paraNames = []
        if re.search(r"{{\w+}}", prompt_template):
            paraNames = re.findall(r"{{.*}}", prompt_template)
            for i in range(len(paraNames)):
                paraNames[i] = paraNames[i][2:-2]
        return paraNames
    def run_Function(self, promptvalue, prenunits ,model):
        ready_prompt = ""
        for value in self.prompt_template[promptvalue]:
            ready_prompt += value[1] + "\n"
        para_name = self.getPromptParams(ready_prompt)
        for index, key in enumerate(para_name):
            ready_prompt = ready_prompt.replace("{{%s}}" % key, prenunits[index])
        Config = LLMConfigurator.Config()
        Config.add_to_config("prompt", ready_prompt)
        if (model["engine"].replace(" ", "") == "DALL-E"):
            response = openai.Image.create(
                prompt=ready_prompt,
                n=1,
                size="512x512",
            )
            image_url = response['data'][0]['url']
            return image_url
        if (model["engine"].replace(" ", "") == "gpt-3.5-turbo"):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": ready_prompt}
                ]
            )
            output = response.choices[0].message["content"]
            return output
        for key in model:
            Config.add_to_config(key, model[key])
        response = openai.Completion.create(
            engine=Config.engine.replace(" ", ""),
            prompt=Config.prompt,
            temperature=float(Config.temperature),
            max_tokens=int(Config.max_tokens),
            top_p=float(Config.top_p),
            frequency_penalty=float(Config.frequency_penalty),
            presence_penalty=float(Config.presence_penalty),
            stop=Config.stop_strs
        )
        output = response["choices"][0]["text"]
        return output
    def run_PythonREPL(self,promptvalue, prenunits, model):
        ready_prompt = ""
        for value in self.prompt_template[promptvalue]:
            ready_prompt += value[1] + "\n"
        para_name = self.getPromptParams(ready_prompt)
        for index, key in enumerate(para_name):
            ready_prompt = ready_prompt.replace("{{%s}}" % key, prenunits[index])
        output = run_python_code(ready_prompt)
        return output
