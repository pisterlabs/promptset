import random
import re
import ast
import pandas as pd
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as message
import pyperclip
from enum import Enum

from keyword_explorer.tkUtils.Buttons import Buttons
from keyword_explorer.tkUtils.TextField import TextField
from keyword_explorer.tkUtils.DataField import DataField
from keyword_explorer.tkUtils.Checkboxes import Checkboxes, Checkbox
from keyword_explorer.tkUtils.TopicComboExt import TopicComboExt
from keyword_explorer.tkUtils.ToolTip import ToolTip
from keyword_explorer.tkUtils.GPT3GeneratorFrame import GPT3GeneratorFrame, GPT3GeneratorSettings
from keyword_explorer.OpenAI.OpenAIEmbeddings import OpenAIEmbeddings

from typing import List, Dict, Pattern

class GPTContextSettings:
    type: str
    prompt: str
    context_prompt:str
    keywords:str
    regex_str:str

    def __init__(self, prompt = "unset", context_prompt = "unset", keywords = "unset", type = "unset", regex_str = "unset"):
        self.prompt = prompt
        self.context_prompt = context_prompt
        self.keywords = keywords
        self.regex_str = regex_str

    def from_dict(self, d:Dict):
        if 'prompt' in d:
            self.prompt = d['prompt']
        if 'context_prompt' in d:
            self.context_prompt = d['context_prompt']
        if 'keywords' in d:
            self.keywords = d['keywords']
        if 'type' in d:
            self.keywords = d['type']
        if 'regex_str' in d:
            self.keywords = d['regex_str']


    def to_dict(self) -> Dict:
        return {'prompt':self.prompt,
                'context_prompt':self.context_prompt,
                'keywords':self.keywords,
                'type':self.type,
                'regex_str':self.regex_str}

class PROMPT_TYPE(Enum):
    def __str__(self):
        return str(self.value)

    NARRATIVE = "Narrative"
    LIST = "List"
    SEQUENCE = "Sequence"
    STORY = "Story"
    QUESTION = "Question"
    TWEET = "Tweet"
    SCIENCE_TWEET = "Science Tweet"
    FACTOID = "Factoid"
    PRESS_RELEASE = "Press Release"
    TWEET_THREAD = "Tweet Thread"


class GPTContextFrame(GPT3GeneratorFrame):
    keyword_filter:DataField
    context_prompt:TextField
    context_text_field:TextField
    prompt_query_cb:Checkbox
    ignore_context_cb:Checkbox
    tab_control:ttk.Notebook
    buttons:Buttons
    project_df:pd.DataFrame
    sequence_response_regex:Pattern
    story_response_reges:Pattern
    list_response_regex:Pattern
    max_tokens:int


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_df = pd.DataFrame()
        self.sequence_response_regex = re.compile(r"\d+[):,.]+|\n+")
        self.list_response_regex = re.compile(r"\d+\W+|\n\d+\W+")
        self.story_response_regex = re.compile(r"\n+")
        self.max_tokens = 512

    def build_frame(self, frm: ttk.Frame, text_width:int, label_width:int):
        row = 0

        self.keyword_filter = DataField(frm, row, "Keywords:", text_width+20, label_width=label_width)
        self.keyword_filter.set_text("Pequod OR Rachel OR Ahab")
        ToolTip(self.keyword_filter.tk_entry, "Keywords (separated by OR) to filter available data")
        row = self.keyword_filter.get_next_row()

        self.context_prompt = TextField(frm, row, "Context:", text_width, height=4, label_width=label_width)
        self.context_prompt.set_text("Working on whaling ships")
        ToolTip(self.context_prompt.tk_text, "The prompt that will provide context")
        row = self.context_prompt.get_next_row()

        self.prompt_text_field = TextField(frm, row, "Prompt:", text_width, height=4, label_width=label_width)
        self.prompt_text_field.set_text("Why is Ahab obsessed with Moby Dick?")
        ToolTip(self.prompt_text_field.tk_text, "The prompt that the GPT will use to generate text from")
        row = self.prompt_text_field.get_next_row()

        cboxes = Checkboxes(frm, row, "Options")
        self.prompt_query_cb = cboxes.add_checkbox("Use Prompt for context", self.handle_checkboxes)
        self.prompt_query_cb.set_val(False)
        self.ignore_context_cb = cboxes.add_checkbox("Ignore context", self.handle_checkboxes)
        self.ignore_context_cb.set_val(False)
        row = cboxes.get_next_row()

        # Add the tabs
        self.tab_control = ttk.Notebook(frm)
        self.tab_control.grid(column=0, row=row, columnspan=2, sticky="nsew")

        gen_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(gen_tab, text='Generated')
        ctx_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(ctx_tab, text='Context')
        src_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(src_tab, text='Sources')
        

        self.response_text_field = TextField(gen_tab, row, 'Response:', text_width, height=11, label_width=label_width)
        ToolTip(self.response_text_field.tk_text, "The response from the GPT will be displayed here")

        self.context_text_field = TextField(ctx_tab, row, 'Context:', text_width, height=11, label_width=label_width)
        ToolTip(self.context_text_field.tk_text, "The context including the prompt")

        self.sources_text_field = TextField(src_tab, row, 'Sources:', text_width, height=11, label_width=label_width)
        ToolTip(self.context_text_field.tk_text, "The sources used for the response")

        row = self.response_text_field.get_next_row()

        self.buttons = Buttons(frm, row, "Actions")
        b = self.buttons.add_button("Ask Question", self.new_prompt_callback, width=-1)
        ToolTip(b, "Gets answer from the GPT")
        b = self.buttons.add_button("Summarize", self.get_summmary_callback, width=-1)
        ToolTip(b, "Gets Summary from the GPT")
        b = self.buttons.add_button(PROMPT_TYPE.NARRATIVE.value, self.get_story_callback, width=-1)
        ToolTip(b, "Gets Story from the GPT")
        b = self.buttons.add_button("Extend", self.extend_callback, width=-1)
        ToolTip(b, "Extends the GPT's response")
        b = self.buttons.add_button("Clear", self.clear_callback, width=-1)
        ToolTip(b, "Clears all the fields")
        b = self.buttons.add_button("Copy", self.clibpboard_callback, width=-1)
        ToolTip(b, "Copies engine, prompt, context, and response to clipboard")
        row = self.buttons.get_next_row()
        self.so.add_object("context_buttons", self.buttons, Buttons)

        self.auto_buttons = Buttons(frm, row, "Automatic")
        b = self.auto_buttons.add_button("Question", self.auto_question_callback, width=-1)
        ToolTip(b, "Randomly selects a level-1 summary and then creates a question based on it")
        b = self.auto_buttons.add_button("Tweet", lambda:self.auto_question_callback(type=PROMPT_TYPE.TWEET), width=-1)
        ToolTip(b, "Randomly selects a level-1 summary and then creates a tweet based on it")
        b = self.auto_buttons.add_button("Science Tweet", lambda:self.auto_question_callback(type=PROMPT_TYPE.SCIENCE_TWEET), width=-1)
        ToolTip(b, "Randomly selects a level-1 summary and then creates a tweet in the style of Science Twitter based on it")
        b = self.auto_buttons.add_button("Thread", lambda:self.auto_question_callback(type=PROMPT_TYPE.TWEET_THREAD), width=-1)
        ToolTip(b, "Randomly selects a level-1 summary and then creates a thread in the style of Science Twitter based on it")
        b = self.auto_buttons.add_button("Factoid", lambda:self.auto_question_callback(type=PROMPT_TYPE.FACTOID), width=-1)
        ToolTip(b, "Randomly selects a level-1 summary and then creates a factoid based on it")
        b = self.auto_buttons.add_button("Press release", lambda:self.auto_question_callback(type=PROMPT_TYPE.PRESS_RELEASE), width=-1)
        ToolTip(b, "Randomly selects a level-1 summary and then creates a press release based on it")
        row = self.auto_buttons.get_next_row()

    def set_project_dataframe(self, df:pd.DataFrame):
        self.project_df = df

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def handle_checkboxes(self, event = None):
        print("prompt_query_cb = {}".format(self.prompt_query_cb.get_val()))
        print("ignore_context_cb = {}".format(self.ignore_context_cb.get_val()))

    def new_prompt_callback(self):
        generate_model_combo:TopicComboExt = self.so.get_object("generate_model_combo")
        model = generate_model_combo.get_text()
        print("using model {}".format(model))
        self.tab_control.select(0)
        self.response_text_field.clear()
        if self.project_df.empty:
            tk.messagebox.showwarning("Warning!", "Please import data first")
            return
        oae = OpenAIEmbeddings()
        ctx_question = self.context_prompt.get_text()
        if self.prompt_query_cb.get_val() or len(ctx_question) < 5:
            ctx_question = self.prompt_text_field.get_text()
        context, origins_list = oae.create_context(ctx_question, self.project_df)

        question = self.prompt_text_field.get_text()
        full_question = question
        if self.ignore_context_cb.get_val() == False:
            full_question = oae.create_question(question=question, context=context)

        self.context_text_field.clear()
        self.context_text_field.set_text(full_question)

        origins = oae.get_origins_text(origins_list)
        self.sources_text_field.clear()
        self.sources_text_field.set_text("\n\n".join(origins))

        self.dp.dprint("Submitting Question: {}".format(question))
        answer = oae.get_response(full_question, model=model, max_tokens=self.max_tokens)
        answer = oae.tk_filter_string(answer)
        self.response_text_field.set_text(answer)

    def auto_question_callback(self, type = PROMPT_TYPE.QUESTION):
        print("GPTContextFrame.auto_question_callback()")
        if self.project_df.empty:
            tk.messagebox.showwarning("Warning!", "Please import data first")
            return

        oae = OpenAIEmbeddings()
        num_lines = 2
        first_line = random.randrange(0, len(self.project_df.index)-num_lines)

        s = self.project_df.iloc[first_line]['parsed_text']
        print("GPTContextFrame.auto_question_callback()\n\tfirst_line id = {} [0 - {}]\n\ttext = {}".format(
            first_line, len(self.project_df.index)-num_lines, s))
        prompt_type = "short question"
        if type == PROMPT_TYPE.TWEET:
            prompt_type = "short tweet"
        elif type == PROMPT_TYPE.SCIENCE_TWEET:
            prompt_type = "short tweet in the style of Science Twitter"
        elif type == PROMPT_TYPE.FACTOID:
            num_lines = 5
            prompt_type = "factoid"
        elif type == PROMPT_TYPE.TWEET_THREAD:
            num_lines = 10
            prompt_type = "science Twitter thread"
        elif type == PROMPT_TYPE.PRESS_RELEASE:
            num_lines = 10
            topic = self.prompt_text_field.get_text()
            if len(topic) < 3:
                topic = "the book Stampede Theory, by Philip Feldman"
                self.prompt_text_field.set_text(topic)
            prompt_type = "press release for {}".format(topic)

        origins_list = []
        context_str = "Create a {} that uses the following context\n\nContext:{}".format(prompt_type, s)
        for i in range(first_line+1, first_line+num_lines, 1):
            series = self.project_df.iloc[i]
            # print("series = {}".format(series))
            s = series['parsed_text']
            context_str += "\n\n###\n\n{}".format(s)
            origin = series['origins']
            origins_list += origin
        context_str += "\n\n{}:".format(prompt_type)
        self.context_text_field.clear()
        self.context_text_field.set_text(context_str)
        origins = oae.get_origins_text(origins_list)
        self.sources_text_field.clear()
        self.sources_text_field.set_text("\n\n".join(origins))
        # print("\tcontext = {}".format(context_str))

        generate_model_combo:TopicComboExt = self.so.get_object("generate_model_combo")
        model = generate_model_combo.get_text()
        print("\tusing model {}".format(model))

        oae = OpenAIEmbeddings()
        question = oae.get_response(context_str, max_tokens=self.max_tokens, model=model)
        question = oae.tk_filter_string(question)
        if type == PROMPT_TYPE.QUESTION:
            self.context_prompt.set_text(question)
            self.prompt_text_field.set_text("{}. Use the style of a white paper.".format(question))
        else:
            self.response_text_field.set_text(question)
        self.tab_control.select(0)



    def get_summmary_callback(self):
        generate_model_combo:TopicComboExt = self.so.get_object("generate_model_combo")
        model = generate_model_combo.get_text()
        print("using model {}".format(model))

        self.tab_control.select(0)
        if self.project_df.empty:
            tk.messagebox.showwarning("Warning!", "Please import data first")
            return
        self.response_text_field.clear()
        oae = OpenAIEmbeddings()
        ctx_prompt = self.context_prompt.get_text()
        if self.prompt_query_cb.get_val():
            ctx_prompt = self.prompt_text_field.get_text()
        context, origins_list = oae.create_context(ctx_prompt, self.project_df)

        question = self.prompt_text_field.get_text()
        full_prompt = oae.create_summary(context=context)

        self.context_text_field.clear()
        self.context_text_field.set_text(full_prompt)

        origins = oae.get_origins_text(origins_list)
        self.sources_text_field.clear()
        self.sources_text_field.set_text("\n\n".join(origins))

        self.dp.dprint("Submitting summary prompt: {}".format(context))
        answer = oae.get_response(full_prompt, max_tokens=self.max_tokens, model=model)
        self.response_text_field.set_text(answer)

    def get_gpt_list(self, oae:OpenAIEmbeddings, ctx_prompt:str, prompt:str, model:str):
        split_regex = re.compile(r"\|+")
        context, origins_list = oae.create_context(ctx_prompt, self.project_df)

        # split the prompt and iterate over the seeds to produce the total output:
        query_list = split_regex.split(prompt)
        template_s = query_list[0].strip()
        response_dict = {}
        for i in range(1, len(query_list)):
            s = query_list[i].strip()
            query_str = template_s.format(s)
            full_prompt = query_str
            self.dp.dprint("Submitting List prompt: {}".format(query_str))
            if self.ignore_context_cb.get_val() == False:
                full_prompt = oae.create_list(prompt=query_str, context=context)
            response = oae.get_response(full_prompt, max_tokens=self.max_tokens, model=model)
            response_dict[query_str] = response

        self.context_text_field.clear()
        self.context_text_field.set_text(full_prompt)

        origins = oae.get_origins_text(origins_list)
        self.sources_text_field.clear()
        self.sources_text_field.set_text("\n\n".join(origins))

        s = ""
        val:str
        for key, val in response_dict.items():
            s += "{}:\n".format(key)
            val_list = self.list_response_regex.split(val)
            for i in range(len(val_list)):
                s += "\t[{}] {}\n".format(i, val_list[i])
        self.response_text_field.set_text(s)

    def get_gpt_sequence(self, oae:OpenAIEmbeddings, ctx_prompt:str, prompt:str, model:str):
        print("GPTContextFrame.get_gpt_sequence()")
        print("\tRaw prompt = {}".format(prompt))
        print("\tContext prompt = {}".format(ctx_prompt))
        print("\tModel = {}".format(model))
        context, origins_list = oae.create_context(ctx_prompt, self.project_df)
        split_regex_1 = re.compile(r"\|+")
        split_regex_2 = re.compile(r"&+")
        query_list = split_regex_1.split(prompt)
        template_s = query_list[0].strip()
        response_dict = {}
        for i in range(1, len(query_list)):
            s = query_list[i].strip()
            s_list = split_regex_2.split(s)
            if len(s_list) == 2:
                s1 = s_list[0].strip()
                s2 = s_list[1].strip()
                if ctx_prompt == prompt:
                    print("\tGetting new context using '{}, {}' ".format(s1, s2))
                    context, origins_list = oae.create_context("{}, {}".format(s1, s2), self.project_df)
                query_str = template_s.format(s1, s2)
                full_prompt = query_str
                if self.ignore_context_cb.get_val() == False:
                    full_prompt = oae.create_sequence(prompt=query_str, context=context)
                print("\tSubmitting Sequence prompt: {}".format(full_prompt))
                response = oae.get_response(full_prompt, max_tokens=self.max_tokens, model=model)

                if len(response) > 3:
                    print("\tStoring response")
                    response_dict[query_str] = response


        self.context_text_field.clear()
        self.context_text_field.set_text(full_prompt)

        origins = oae.get_origins_text(origins_list)
        self.sources_text_field.clear()
        self.sources_text_field.set_text("\n\n".join(origins))

        s = ""
        val:str
        for key, val in response_dict.items():
            s += "\n{}:\n".format(key)
            val_list = self.sequence_response_regex.split(val)
            for i in range(len(val_list)):
                val_str = val_list[i].strip()
                if len(val_str) > 2:
                    s += "\t[{}] {}\n".format(i, val_str)
        self.response_text_field.set_text(s)

    def get_gpt_story(self, oae:OpenAIEmbeddings, ctx_prompt:str, prompt:str, model:str):
        context, origins_list = oae.create_context(ctx_prompt, self.project_df)
        full_prompt = prompt
        if self.ignore_context_cb.get_val() == False:
            full_prompt = oae.create_narrative(prompt=prompt, context=context)

        self.context_text_field.clear()
        self.context_text_field.set_text(full_prompt)

        origins = oae.get_origins_text(origins_list)
        self.sources_text_field.clear()
        self.sources_text_field.set_text("\n\n".join(origins))

        self.dp.dprint("Submitting Story prompt: {}".format(prompt))
        response = oae.get_response(full_prompt, max_tokens=self.max_tokens, model=model)
        response_list = self.story_response_regex.split(response)
        s = "{}".format(prompt)
        for i in range(len(response_list)):
            response = response_list[i]
            s = "{}\n[P {}]: {}\n".format(s, i, response)
        self.response_text_field.clear()
        self.response_text_field.set_text(s)

    def get_story_callback(self):
        generate_model_combo:TopicComboExt = self.so.get_object("generate_model_combo")
        model = generate_model_combo.get_text()
        print("using model {}".format(model))

        self.tab_control.select(0)
        if self.project_df.empty:
            tk.messagebox.showwarning("Warning!", "Please import data first")
            return

        prompt_type = self.buttons.get_button_label(PROMPT_TYPE.NARRATIVE.value)
        self.response_text_field.clear()
        oae = OpenAIEmbeddings()
        ctx_prompt = self.context_prompt.get_text()
        if self.prompt_query_cb.get_val() or len(ctx_prompt) < 3:
            ctx_prompt = self.prompt_text_field.get_text()

        prompt = self.prompt_text_field.get_text()
        if prompt_type == PROMPT_TYPE.NARRATIVE.value:
            self.get_gpt_story(oae, ctx_prompt, prompt, model)
        elif prompt_type == PROMPT_TYPE.LIST.value:
            self.get_gpt_list(oae, ctx_prompt, prompt, model)
        elif prompt_type == PROMPT_TYPE.SEQUENCE.value:
            self.get_gpt_sequence(oae, ctx_prompt, prompt, model)

    def extend_callback(self):
        generate_model_combo:TopicComboExt = self.so.get_object("generate_model_combo")
        model = generate_model_combo.get_text()
        print("using model {}".format(model))

        if self.project_df.empty:
            tk.messagebox.showwarning("Warning!", "Please import data first")
            return
        oae = OpenAIEmbeddings()
        prompt = "{} {}".format(self.prompt_text_field.get_text(), self.response_text_field.get_text())
        self.prompt_text_field.clear()
        self.prompt_text_field.set_text(prompt)
        self.response_text_field.clear()
        self.dp.dprint("Submitting extend prompt:")
        response = oae.get_response(prompt, model=model)
        self.response_text_field.set_text(response)

    def clear_callback(self, clear_keywords = True):
        if clear_keywords:
            self.keyword_filter.clear()
        self.context_prompt.clear()
        self.prompt_text_field.clear()
        self.response_text_field.clear()
        self.context_text_field.clear()
        self.sources_text_field.clear()

    def clibpboard_callback(self):
        print("clibpboard_callback")
        generate_model_combo:TopicComboExt = self.so.get_object("generate_model_combo")
        context = self.context_prompt.get_text()
        prompt = self.prompt_text_field.get_text()
        if self.prompt_query_cb.get_val() or len(context) < 5:
            context = prompt
        s = "Model: {}\n\nContext: {}\n\nPrompt: {}\n\nResponse: {}".format(
            generate_model_combo.get_text(), context, prompt, self. response_text_field.get_text())
        pyperclip.copy(s)
        pyperclip.paste()
        print("\tPased {} to clipboard".format(s))

    def set_params(self, settings:GPTContextSettings):
        self.prompt_text_field.clear()
        self.context_prompt.clear()
        self.keyword_filter.clear()

        self.prompt_text_field.set_text(settings.prompt)
        self.context_prompt.set_text(settings.context_prompt)
        self.keyword_filter.set_text(settings.keywords)

    def get_settings(self) -> GPTContextSettings:
        gs = GPTContextSettings()
        gs.prompt = self.prompt_text_field.get_text()
        gs.context_prompt = self.context_prompt.get_text()
        gs.keywords = self.keyword_filter.get_text()
        return gs
