import omni.ext
import omni.ui as ui
#create a file apikeys.py in the same folder as extension.py and add 2 variables:
# API_KEY: "your openai api key"
# PYTHON_PATH: "the path of the python folder where the openai python library is installed"
from .apikeys import apikey
from .apikeys import pythonpath
import pyperclip

import sys
sys.path.append(pythonpath)
import openai

#tokens used in the OpenAI API response
openaitokensresponse = 40


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[omni.openai.snippet] MyExtension startup")


        self._window = ui.Window("OpenAI GPT-3 Text Generator", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
               
                
                prompt_label = ui.Label("Your Prompt:")
                prompt_field = ui.StringField(multiline=True)

                result_label = ui.Label("OpenAI GPT-3 Result:")
                label_style = {"Label": {"font_size": 16, "color": 0xFF00FF00,}}
                result_actual_label = ui.Label("The OpenAI generated text will show up here", style=label_style, word_wrap=True)
        

                def on_click():
                    # Load your API key from an environment variable or secret management service
                    #openai.api_key = "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"
                    openai.api_key = apikey

                    my_prompt = prompt_field.model.get_value_as_string().replace("\n", " ")
                    
                   
                    

                    response = openai.Completion.create(engine="text-davinci-001", prompt=my_prompt, max_tokens=openaitokensresponse)

                    #parse response as json and extract text
                    text = response["choices"][0]["text"]
                    pyperclip.copy(text)
                    result_actual_label.text = ""
                    result_actual_label.text = text
                   
                ui.Button("Generate and Copy to Clipboard", clicked_fn=lambda: on_click())

    def on_shutdown(self):
        print("[omni.openai.snippet] MyExtension shutdown")
