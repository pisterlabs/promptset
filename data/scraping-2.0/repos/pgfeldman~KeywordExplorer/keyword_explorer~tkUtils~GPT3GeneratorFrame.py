import re
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as message
from datetime import datetime
from tkinter import filedialog

from keyword_explorer.tkUtils.ConsoleDprint import ConsoleDprint
from keyword_explorer.tkUtils.Buttons import Buttons
from keyword_explorer.tkUtils.ToolTip import ToolTip
from keyword_explorer.tkUtils.TextField import TextField
from keyword_explorer.tkUtils.DataField import DataField
from keyword_explorer.tkUtils.TopicComboExt import TopicComboExt
from keyword_explorer.tkUtils.LabeledParam import LabeledParam
from keyword_explorer.OpenAI.OpenAIComms import OpenAIComms
from keyword_explorer.utils.SharedObjects import SharedObjects


from typing import List, Dict, Callable

class GPT3GeneratorSettings:
    prompt:str
    model:str
    tokens:int
    temperature:float
    presence_penalty:float
    frequency_penalty:float
    auto_runs:int

    def __init__(self, prompt = "unset", model = "text-davinci-003", tokens = 64,
                 temperature = 0, presence_penalty = 0.8, frequency_penalty = 0,
                 auto_runs = 10):
        self.prompt = prompt
        self.model = model
        self.tokens = tokens
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.auto_runs = auto_runs

    def from_dict(self, d:Dict):
        if 'probe_str' in d:
            self.prompt = d['probe_str']
        if 'prompt' in d:
            self.prompt = d['prompt']
        if 'generate_model' in d:
            self.model = d['generate_model']
        if 'tokens' in d:
            self.tokens = d['tokens']
        if 'temp' in d:
            self.temperature = d['temp']
        if 'presence_penalty' in d:
            self.presence_penalty = d['presence_penalty']
        if 'frequency_penalty' in d:
            self.frequency_penalty = d['frequency_penalty']
        if 'automated_runs' in d:
            self.auto_runs = d['automated_runs']

class GPT3GeneratorFrame:
    oai: OpenAIComms
    dp:ConsoleDprint
    so:SharedObjects
    generate_model_combo: TopicComboExt
    prompt_text_field:TextField
    response_text_field:TextField
    tokens_param: LabeledParam
    temp_param: LabeledParam
    presence_param: LabeledParam
    frequency_param: LabeledParam
    regex_field:DataField
    auto_field:DataField
    buttons:Buttons
    saved_prompt_text:str
    saved_response_text:str

    def __init__(self, oai:OpenAIComms, dp:ConsoleDprint, so:SharedObjects):
        self.oai = oai
        self.dp = dp
        self.so = so

    def build_frame(self, frm: ttk.Frame, text_width:int, label_width:int):
        engine_list = self.oai.list_models(exclude_list = [":", "ada", "embed", "similarity", "code", "edit", "search", "audio", "instruct", "2020", "if", "insert", "whisper"])
        engine_list = sorted(engine_list, reverse=True)
        row = 0
        self.generate_model_combo = TopicComboExt(frm, row, "Model:", self.dp, entry_width=25, combo_width=25)
        self.generate_model_combo.set_combo_list(engine_list)
        self.generate_model_combo.set_text(engine_list[0])
        self.generate_model_combo.tk_combo.current(0)
        ToolTip(self.generate_model_combo.tk_combo, "The GPT-3 model used to generate text")

        row = self.generate_model_combo.get_next_row()
        row = self.build_generate_params(frm, row)

        self.prompt_text_field = TextField(frm, row, "Prompt:", text_width, height=6, label_width=label_width)
        self.prompt_text_field.set_text("Once upon a time there was")
        ToolTip(self.prompt_text_field.tk_text, "The prompt that the GPT will use to generate text from")
        row = self.prompt_text_field.get_next_row()

        self.response_text_field = TextField(frm, row, 'Response:', text_width, height=11, label_width=label_width)
        ToolTip(self.response_text_field.tk_text, "The response from the GPT will be displayed here")
        row = self.response_text_field.get_next_row()

        self.regex_field = DataField(frm, row, 'Parse regex:', text_width, label_width=label_width)
        self.regex_field.set_text(r"\n|[\.!?] |([\.!?]\")")
        ToolTip(self.regex_field.tk_entry, "The regex used to parse the GPT response. Editable")
        row = self.regex_field.get_next_row()

        self.auto_field = DataField(frm, row, 'Run count:', text_width, label_width=label_width)
        self.auto_field.set_text("10")
        ToolTip(self.auto_field.tk_entry, "The number of times the prompt will be run by 'Automate'")
        row = self.auto_field.get_next_row()

        self.buttons = Buttons(frm, row, "Actions")
        b = self.buttons.add_button("Generate", self.new_prompt_callback)
        ToolTip(b, "Sends the prompt to the GPT")
        b = self.buttons.add_button("Add", self.extend_prompt_callback)
        ToolTip(b, "Adds the response to the prompt")
        b = self.buttons.add_button("Parse", self.parse_response_callback)
        ToolTip(b, "Parses the response into a list for embeddings")

    def add_button(self, label:str, callback:Callable, tooltip:str):
        b = self.buttons.add_button(label, callback)
        ToolTip(b, tooltip)

    def build_generate_params(self, parent:tk.Frame, row:int) -> int:
        f = tk.Frame(parent)
        f.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=1, pady=1)

        self.tokens_param = LabeledParam(f, 0, "Tokens:")
        self.tokens_param.set_text('256')
        ToolTip(self.tokens_param.tk_entry, "The number of tokens the model will generate")

        self.temp_param = LabeledParam(f, 2, "Temp:")
        self.temp_param.set_text('0.7')
        ToolTip(self.temp_param.tk_entry, "The randomness of the response (0.0 - 1.0)")

        self.presence_param = LabeledParam(f, 4, "Presence penalty:")
        self.presence_param.set_text('0.3')
        ToolTip(self.presence_param.tk_entry, "Increases liklihood of new topics")

        self.frequency_param = LabeledParam(f, 6, "Frequency penalty:")
        self.frequency_param.set_text('0.3')
        ToolTip(self.frequency_param.tk_entry, "Suppresses repeating text")
        return row + 1

    def set_params(self, settings:GPT3GeneratorSettings):
        self.prompt_text_field.clear()
        self.auto_field.clear()
        self.tokens_param.clear()
        self.generate_model_combo.clear()
        self.temp_param.clear()
        self.presence_param.clear()
        self.frequency_param.clear()

        self.tokens_param.set_text(str(settings.tokens))
        self.temp_param.set_text(str(settings.temperature))
        self.presence_param.set_text(str(settings.presence_penalty))
        self.frequency_param.set_text(str(settings.frequency_penalty))
        self.generate_model_combo.set_text(settings.model)
        self.prompt_text_field.set_text(settings.prompt)
        self.auto_field.set_text(str(settings.auto_runs))

    def set_prompt(self, prompt:str):
        self.prompt_text_field.set_text(prompt)

    def get_settings(self) -> GPT3GeneratorSettings:
        gs = GPT3GeneratorSettings()
        gs.model = self.generate_model_combo.get_text()
        gs.prompt = self.prompt_text_field.get_text()
        gs.tokens = self.tokens_param.get_as_int()
        gs.temperature = self.temp_param.get_as_float()
        gs.presence_penalty = self.presence_param.get_as_float()
        gs.frequency_penalty = self.frequency_param.get_as_float()
        gs.auto_runs = self.auto_field.get_as_int()
        return gs

    def get_gpt3_response(self, prompt:str) -> str:
        """
        Method that takes a prompt and gets the response back via the OpenAI API
        :param prompt: The prompt to be sent to the GPT-3
        :return: The GPT-3 Response
        """
        if len(prompt) < 3:
            self.dp.dprint("get_gpt3_response() Error. Prompt too short: '{}'".format(prompt))
            return ""

        # print(prompt)
        self.oai.max_tokens = self.tokens_param.get_as_int()
        self.oai.temperature = self.temp_param.get_as_float()
        self.oai.frequency_penalty = self.frequency_param.get_as_float()
        self.oai.presence_penalty = self.presence_param.get_as_float()
        self.oai.engine = self.generate_model_combo.get_text()

        results = self.oai.get_prompt_result(prompt, False)
        self.dp.dprint("\n------------\ntokens = {}, engine = {}\nprompt = {}".format(self.oai.max_tokens, self.oai.engine, prompt))

        # clean up before returning
        s = results[0].strip()
        self.dp.dprint("gpt_response: {}".format(s))
        return s

    def new_prompt_callback(self):
        split_regex = re.compile(r"[\n]+")
        prompt = self.prompt_text_field.get_text()
        response = self.get_gpt3_response(prompt)
        l = split_regex.split(response)
        response = "\n".join(l)
        self.response_text_field.set_text(response)

    def extend_prompt_callback(self):
        prompt = "{} {}".format(self.prompt_text_field.get_text(), self.response_text_field.get_text())
        self.prompt_text_field.set_text(prompt)
        self.response_text_field.clear()

    def get_list(self, to_parse:str, regex_str:str = ",") -> List:
        rlist = re.split(regex_str, to_parse)
        to_return = []
        for t in rlist:
            if t != None:
                to_return.append(t.strip())
        to_return = [x for x in to_return if x] # filter out the blanks
        return to_return

    def parse_response_callback(self):
        # get the regex
        split_regex = self.regex_field.get_text()

        # get the prompt and respnse text blocks
        self.saved_prompt_text = self.prompt_text_field.get_text()
        self.saved_response_text = self.response_text_field.get_text()
        full_text = self.saved_response_text

        # build the list of parsed text
        self.parsed_full_text_list = self.get_list(full_text, split_regex)
        # print(response_list)

        if len(self.parsed_full_text_list) > 1:
            count = 0
            for r in self.parsed_full_text_list:
                if len(r) > 1:
                    self.dp.dprint("line {}: {}".format(count, r))
                    count += 1
        else:
            message.showwarning("Parse Error",
                                "Could not parse [{}]".format(self.response_text_field.get_text()))

    # make this a "restore" button?
    def test_data_callback(self):
        prompt_text = '''Once upon a time there was a man who had been a soldier, and who had fought in the wars. After some years he became tired of fighting, and he stopped his soldiering and went away to live by himself in the mountains. He built a hut for himself, and there he lived for many years. At last one day there was a knocking at his door. He opened it and found no one there.

The next day, and the next, and the next after that there was a knocking at his door, but when he opened it no one was ever there.

At last he got so cross that he could not keep away from home any more than usual. When he opened the door and found no one there, he was so angry that he threw a great stone after whoever it was that knocked.

Presently a voice called out to him and said: “I am coming back soon again; you must be careful not to throw stones at me then”; but the voice did not say who it was that spoke.

The second time the man’s heart failed him as soon as he opened his door; but when he heard the voice saying: “Be careful not to throw stones this time,” he felt quite sure that'''
        response_text = '''it was the same voice. Then he knew that it was his Guardian Spirit that spoke to him.

The third time the man was not afraid, but as soon as he opened the door and saw no one, he threw stones at it.

Then a great storm arose and the thunder rolled among the mountains, and the lightning flashed in his eyes and blinded him, and all about him there were voices shouting: “It is your Guardian Spirit that you have killed!”

And when he could see again, he looked up and saw that the hut had disappeared and that in its place stood a dark pine-tree. He ran to look for his hut, but it was nowhere to be found; he looked up and down the valley, but there was no sign of it anywhere. He called out loudly for his hut to come back,—but it never came back again. The hut had become a big pine-tree, and even the Guardian Spirit could not make it come back again.'''
        self.prompt_text_field.set_text(prompt_text)
        self.response_text_field.set_text(response_text)