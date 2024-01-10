import sys
from copy import deepcopy
import codecs
import openai
import tkinter as tk
import tkinter.ttk as ttk
import os
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
from pathlib import Path
from time import localtime
from glob import glob
from fpdf import FPDF
import fpdf
from time import sleep

from promptdesignerdataset.dataset import PromptDesignerDataset, LoadJsonData, SaveJsonData
from promptdesigner.Answers import AnswersInterface
from promptdesigner.Export import ExportSessionData

#  messages

_license = """
MIT License

Copyright (c) 2022-present Patrizio Bellan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

_help_questions_list = """

QUESTION_BLOCK = '==='
    COMMENT_BLOCK = '#'


"""

_help_legend = """
Symbols Legend:
---------------
'#'    -> commment a line
'==='  -> question block deliminter
'FILE:'-> read the content of a file
'+++'  -> parametrized question block delimiter
'@'    -> parameter assign symbol
"""

_help_parametrized_question = """
How to write a parametrized question or a set of questions.

As you do to write a set of questions in a single questions-list file,
each parametrized question must be written within a parameter-block.
A block always begin and end with a line contaning the delimiter '+++' (three plus simble).
within the block you must set a value for each parameter key you defined in your prompt.
To set a value for a key you must follow the following schema:
        key-name @ key-value
You can also assign the content of a file to a parameter by using
the key-word FILE: and the path to the file. 
For example, to assign "5" to the parameter (key-name) "a" and the content
of the file "b.txt" to the parameter "b" you should write your parametrized-ql
as shown:
        a @ 5
        b @ FILE:'b.txt'

Another example.
Considering the following prompt file.
        prompt: "is {x} greater than {y}?"        
This prompt has two parameters-key.
So, you must provide a value for each parameter in your questions_list file.
You can set one or more different settings for a prompt in the same file.
You can just set a parameter once in a block its key-value pair will be used in all the other blocks.
It is important that each setting is within a parameters-block
        questions_list = "
+++     
x @ 5
y @ 3
+++
+++
x @ me
y @ you
+++
+++
y @ -1
+++
+++
x @FILE:file_containing_x.whatever
y @ 3
+++
                        "
This setting gets four configurations:
1- x = 5, y = 5
2- x = me, y = you 
3- x = me, y = -1  
4- x = 'the content of the file "file_conrtaining_x.whatever', y = 3
"""

_temperature_message = """
temperature:
What sampling temperature to use. Higher values means the model will take more risks.
Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
"""
_nucleus_message = """
An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the 
tokens with top_p probability mass. 
So 0.1 means only the tokens comprising the top 10% probability mass are considered.
"""
_presence_penalty_message = """
Number between -2.0 and 2.0. 
Positive values penalize new tokens based on whether they appear in the text so far, 
increasing the model's likelihood to talk about new topics.
"""
_frequency_penalty_message = """
Number between -2.0 and 2.0. 
Positive values penalize new tokens based on their existing frequency in the text so far, 
decreasing the model's likelihood to repeat the same line verbatim."""

_engine_message = """
Davinci
--------------------
Davinci is the most capable engine and can perform any task the other models can perform and often with less instruction. For applications requiring a lot of understanding of the content, like summarization for a specific audience and creative content generation, Davinci is going to produce the best results. These increased capabilities require more compute resources, so Davinci costs more per API call and is not as fast as  the other engines. Another area where Davinci shines is in understanding the intent of text. Davinci is quite good at solving many kinds of logic problems and explaining the motives of characters. Davinci has been able to solve some of the most challenging AI problems involving cause and effect.
Good at: Complex intent, cause and effect, summarization for audience

Curie
--------------------
Curie is extremely powerful, yet very fast. While Davinci is stronger when it comes to analyzing complicated text, 
Curie is quite capable for many nuanced tasks like sentiment classification and summarization. Curie is also quite good at answering questions and performing Q&A and as a general service chatbot.
Good at: Language translation, complex classification, text sentiment, summarization

Babbage
--------------------
Babbage can perform straightforward tasks like simple classification. It’s also quite capable when it comes to Semantic 
Search ranking how well documents match up with search queries.
Good at: Moderate classification, semantic search classification

Ada
--------------------
Ada is usually the fastest model and can perform tasks like parsing text, address correction and certain kinds of classification tasks that don’t require too much nuance. Ada’s performance can often be improved by providing more context.
Good at: Parsing text, simple classification, address correction, keywords
Note: Any task performed by a faster model like Ada can be performed by a more powerful model like Curie or Davinci.
"""

class ToolTipParameters:

    def __init__(self, widget, message):
        self.message = message

        self.widget = widget

        self.widget.bind('<Enter>', self.on_enter)
        self.widget.bind('<Leave>', self.on_leave)

    def on_enter(self, event):
        self.tooltip = tk.Toplevel()
        self.tooltip.overrideredirect(True)
        self.tooltip.geometry(f'+{event.x_root + 30}+{event.y_root + 10}')
        tk.Label(self.tooltip,
                 text=self.message,
                 anchor='w',
                 justify='left',
                 wraplength=500,
                 bg='white').pack(fill='both')
        self.tooltip.after(5000, self.on_leave)

    def on_leave(self, *event):
        self.tooltip.destroy()


class ToolTipQuestionsList:
    tooltip = None
    def __init__(self, widget, message):
        self.message = message

        self.widget = widget

        self.widget.bind('<Enter>', self.on_enter)
        self.widget.bind('<Leave>', self.on_leave)

    def on_enter(self, event):
        self.tooltip = tk.Toplevel()
        self.tooltip.overrideredirect(True)
        # self.tooltip.geometry(f'+{event.x_root + 30}+{event.y_root + 10}')
        self.tooltip.geometry(f'+{event.x_root}+{event.y_root}')
        tk.Label(self.tooltip,
                 text=self.message,
                 anchor='w',
                 justify='left',
                 wraplength=500,
                 bg='white').pack(fill='both')
        self.tooltip.bind('<FocusOut>', self.on_leave)
        self.tooltip.after(15000, self.on_leave)

    def on_leave(self, *event):
        self.tooltip.destroy()


class PromptDesigner(tk.Frame):
    __version__ = '1.3.3'
    PADX = 5
    PADY = 2

    GPT_STATUS_NOT_READY = ':('
    GPT_STATUS_READY = ':)'
    GPT_STATUS_ASKING = ':|'
    # _ANSWER_PATTERN_HEADER = 'T={temperature}|{prompt}'
    _ANSWER_PATTERN_HEADER = 'Engine: {engine} | prompt name: {prompt} - Temperature: {temperature}, Nucleus:{nucleus}, Presence Penalty: {presence_penalty}, Frequency Penalty: {frequency_penalty}'

    _ANSWER_PATTERN_SINGLE = '\tA: {answer}'
    _ANSWER_FILLER = ''  # '=' * 80
    SAMPLING_TEMPERATURE = 'temperature-sampling'
    SAMPLING_NUCLEUS = 'nucleus-sampling'
    ASK_QUESTIONS_LIST = 'ask-questions-list'
    ASK_SINGLE_QUESTION = 'ask-single-question'

    DEFAULT_STOPWORDS = ["Q:", "###", "<|endoftext|>"]
    ENGINES_LIST = [
                    'text-davinci-001',
                    'text-curie-001',
                    'text-babbage-001',
                    'text-ada-001',

                    'davinci',
                    'curie',
                    'babbage',
                    'ada']

    EMPTY_QUESTIONS_LIST = '# this is a comment.\nthis is not. \n\n===\nthis is a block.\n so a single question splitted over multiple\nlines\n=== \n\nthis is another question outside the block'
    EMPTY_COMMENT = '# add a comment to this prompt'
    api_keys_filename = 'keys.json'
    #
    # _OPTION_MENU_EXPANDED = 220
    # _OPTION_MENU_CLOSED = 1

    SETTING_API_KEY = 'api-key'
    SETTING_ANSWERDATA_FILENAME = 'settings-answerdata-filename'
    SETTING_PROMPT_NAME = 'settings-prompt-name'
    SETTING_QUESTION_TYPE = 'settings-question-type'
    SETTING_QUESTION_LIST = 'settings-question-list'

    SETTINGS_FILENAME = 'settings.json'

    # FILE_TYPE_ANSWER_DATA = (("answer data", "*.answer-data"), ("all files", "*.*"))
    FILE_TYPE_QUESTIONS_LIST = (("questions-list", "*.questions-list"), ("all files", "*.*"))
    FILE_TYPE_PROMPT = (("prompt file", "*.prompt"), ("all files", "*.*"))
    FILE_TYPE_SESSION = (("session file", "*.session"), ("all files", "*.*"))

    #############################

    QUESTION_TAG = 'Q: '
    QUESTION_BLOCK = '==='
    COMMENT_BLOCK = '#'

    EXPORT_PROMPT_COMMENT_BEGIN = '=== COMMENT ===='
    EXPORT_PROMPT_PROMPT_BEGIN = '=== PROMPT ==='

    ### PARAMETRIZED QUESTIONS ###
    PARAMETRIZED_QUESTION_PARAMETERS_DELIM = '+++'
    PARAMETRIZED_QUESTION_PARAMETERS_ASSIGN_SYMBOL = '@'
    PARAMETRIZED_QUESTION_PARAMETERS_FILE_CONTENT = 'FILE:'
    ##############################

    def AskGPT3(self,
                parameters):
        # from pprint import pprint
        # pprint(parameters)
        # print('status: dev')
        # print('AskGPT3 disabled')
        # return None
        # print(parameters)
        try:
            response = openai.Completion.create(
                engine=parameters[self.data.ENGINE],  # "text-davinci-001",
                prompt=parameters[self.data.PROMPT_TEXT],  # '',
                max_tokens=int(parameters[self.data.MAX_TOKEN]),  # 150,
                stop=parameters[self.data.STOPWORDS_LIST],  # ["\n", "<|endoftext|>"],
                temperature=parameters[self.data.TEMPERATURE],
                top_p=parameters[self.data.NUCLEUS],  # nucleus sampling
                n=parameters[self.data.N_],  # n_
                presence_penalty=parameters[self.data.PRESENCE_PENALTY],
                frequency_penalty=parameters[self.data.FREQUENCY_PENALTY],
                # logprobs=1,
            )
            return deepcopy(response)
        except Exception as err:
            messagebox.showerror('',
                                 str(err))
            raise err


    def __init__(self,
                 parent,
                 data=None):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.data = data or PromptDesignerDataset()

        self.parent.title('Prompt Designer')

        path = Path(__file__).absolute()
        logofile = path.parent.joinpath('prompt-designer-icon.png')
        logofile = str(logofile.absolute())
        self.logo_ = tk.PhotoImage(file=logofile, master=self.parent)
        self.parent.tk.call('wm', 'iconphoto', self.parent._w, self.logo_)

        self.parent.bind('<Command-,>', self.ShowParametersSettings)
        self.parent.bind('<Command-m>', self.ShowCommentPrompt)
        self.parent.bind('<Command-l>', self.ShowEditQuestionsList)
        self.parent.bind('<Command-e>', self.ExportPdf_QuestionsListAndAnswers)

        self.parent.bind('<Command-d>', self._duplicate_prompt)
        self.parent.bind('<Command-f>', self._duplicate_questions_list)

        self.parent.bind('<Command-c>', self._clear_session)
        self.__init_variables__()
        self._create_frame()
        # self._load_questions_list()
        # self._load_prompt_names()

        # self.sampling_technique.set(self.SAMPLING_TEMPERATURE)
        self._avoid_asking_same_question.set(True)
        self.question_type_of_questions.set(self.ASK_SINGLE_QUESTION)

        self.prompt_name_str.trace_variable('w', self._show_prompt)
        self.api_key.trace_variable('w', self._set_api_key)
        self.prompt_retrieved.configure(font=("Times New Roman", 15))
        self.session.configure(font=("Times New Roman", 15,))


        self._load_default_setting()
        self._load_questions_list()
        self._load_prompt_names()

    def close_all_answers_objects(self, *event):
        for sess in self.sessions_answer_objects:
            sess.destroy()
        self.sessions_answer_objects.clear()

    def __init_variables__(self):
        #  TopLevel Frames
        self.toplevel_settings = None
        self.toplevel_ql = None
        #  record answer objects opened
        self.sessions_answer_objects = list()

        self.default_path = '../src'
                            # self.default_path
        # self.prompt_to_retrieve_string = tk.StringVar()
        # self.question_to_ask_string = tk.StringVar()
        self.record_json_status = tk.BooleanVar()
        self.record_json_status.set(True)

        self.gpt_status = tk.StringVar()
        self.prompt_name_str = tk.StringVar()
        self.temperature_str = tk.StringVar()
        self.nucleus_str = tk.StringVar()
        self.questions_list_name = tk.StringVar()
        self.question_type_of_questions = tk.StringVar()

        # self.sampling_technique = tk.StringVar()
        self.nucleus_sampling_status = tk.BooleanVar()
        self.temperature_sampling_status = tk.BooleanVar()

        self._avoid_asking_same_question = tk.BooleanVar()

        self.presence_penalty_str = tk.StringVar()
        self.presence_penalty_status = tk.BooleanVar()
        self.frequency_penalty_str = tk.StringVar()
        self.frequency_penalty_status = tk.BooleanVar()
        # self.stopwords = tk.Variable()

        self.api_keys = dict()
        self.api_key = tk.StringVar()
        self.api_key.trace_variable('w', self._set_api_key)

        self._shared_questions_list_str = tk.Variable()
        self._shared_comment_prompt_str = tk.Variable()
        self._shared_ask_single_question_str = tk.Variable()

        self.engine_str = tk.StringVar()
        self.n_str = tk.IntVar()
        self.max_tokens = tk.DoubleVar()
        self.data_time = None
        self.stopword = tk.StringVar()
        #  text of the button
        self.option_text_str = tk.StringVar()
        #  handle status of option menu frame
        self._option_menu_status = False

    def _load_api_keys(self):
        self.api_keys = LoadJsonData(self.api_keys_filename)
        return self.api_keys

    def _set_api_key(self, *event):
        if self.api_keys:
            key = self.api_keys[self.api_key.get()]
            openai.api_key = key

    def _set_gpt_status_not_ready(self):
        self.gpt_status.set(self.GPT_STATUS_NOT_READY)
        self.trafic_light.configure({'bg': 'red'})

    def _set_gpt_status_ready(self):
        self.gpt_status.set(self.GPT_STATUS_READY)
        self.trafic_light.configure({'bg': 'green'})

    def _set_gpt_status_asking(self):
        self.gpt_status.set(self.GPT_STATUS_ASKING)
        self.trafic_light.configure({'bg': 'blue'})

    def _clear_session(self, *event):
        self.session.delete(0, 'end')
        self.data.UpdateSession('')
        self.data.UpdateData()

    def _create_menu(self):
        menubar = tk.Menu(self.parent)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Create a new answer-dataset", command=self._create_new_answers_data)
        filemenu.add_command(label="Open an answer-dataset", command=self.load_answers_data)
        filemenu.add_command(label="Save As", command=self.save_as_answers_data)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.parent.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        parameters = tk.Menu(menubar, tearoff=0)
        parameters.add_command(label="Set As Default setting", command=self._set_as_default_setting)
        parameters.add_separator()
        parameters.add_command(label="Set Parameters", command=self.ShowParametersSettings)
        parameters.add_separator()
        parameters.add_command(label="Handle Keys", command=self.ShowHandleApiKeys)
        menubar.add_cascade(label="Settings", menu=parameters)

        prompt = tk.Menu(menubar, tearoff=0)
        prompt.add_command(label="Add comment to prompt", command=self.ShowCommentPrompt)
        prompt.add_separator()
        prompt.add_command(label="Show Composed Prompt", command=self._show_composed_prompt)
        prompt.add_command(label="Duplicate Prompt", command=self._duplicate_prompt)
        prompt.add_separator()
        prompt.add_command(label="Import Prompt", command=self.import_prompt)
        prompt.add_command(label="Import Prompts", command=self.import_prompts)
        prompt.add_separator()
        prompt.add_command(label="Export Prompt", command=self._export_prompt)
        prompt.add_command(label="Export Prompts", command=self._export_prompts)
        prompt.add_separator()
        prompt.add_command(label='Delete Prompt', command=self._delete_prompt)
        menubar.add_cascade(label="Prompt", menu=prompt)

        questionlist = tk.Menu(menubar, tearoff=0)
        questionlist.add_command(label="Duplicate Questions list", command=self._duplicate_questions_list)
        questionlist.add_command(label="Import Questions list", command=self.import_questions_list)
        questionlist.add_command(label="Import Questions listS", command=self.import_questions_lists)
        questionlist.add_separator()
        questionlist.add_command(label="Export Questions list", command=self._export_question_list)
        questionlist.add_command(label="Export Questions listS", command=self._export_question_lists)
        questionlist.add_separator()
        questionlist.add_command(label="Export Questions list and Answers",
                                 command=self.ExportPdf_QuestionsListAndAnswers,
                                 )
        questionlist.add_separator()
        questionlist.add_command(label="Delete Questions list",
                                 command=self._delete_questions_list)

        menubar.add_cascade(label="Questions list", menu=questionlist)

        answers = tk.Menu(menubar, tearoff=0)
        answers.add_command(label="Show Answers", command=self._show_answers_data)
        answers.add_command(label="Export Answers", command=self._export_answer_data)
        menubar.add_cascade(label="Answers", menu=answers)

        session = tk.Menu(menubar, tearoff=0)
        session.add_command(label="Clear Session", command=self._clear_session)
        session.add_command(label="Export Session", command=self._export_session)
        session.add_command(label="Close answer windows", command=self.close_all_answers_objects)
        menubar.add_cascade(label="Session", menu=session)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="take a tour", command=None)
        helpmenu.add_command(label="Version", command=self._show_version)
        helpmenu.add_command(label="Credits", command=self._show_credit)
        helpmenu.add_command(label="License", command=self._show_license)
        menubar.add_cascade(label="help", menu=helpmenu)

        self.parent.config(menu=menubar)

    def _show_version(self, *event):
        messagebox.showinfo('',
                            'version: {}'.format(self.__version__))
    def _show_credit(self, *event):
        messagebox.showinfo('',
                            'Developed by Patrizio Bellan\n PhD student FBK-UniBz')
    def _show_license(self, *event):
        messagebox.showinfo('',
                            _license)

    def _create_frame(self):
        padx = self.PADX  # (self.PADX, self.PADX)
        pady = self.PADY  # (self.PADY, self.PADY)

        self._create_menu()
        self.frame = tk.Frame(self.parent,
                              padx=padx, pady=pady)
        self.frame.pack(side='right',
                        fill='both',
                        expand=True)
        # self._frame_parameters_settings = tk.Frame(self.parent, bg='blue')
        # self._frame_parameters_settings.pack(side='left', fill='y', expand=True)
        # self._frame_parameters_settings.propagate(0)
        # self.ShowParametersSettings()

        self.frame_body = tk.Frame(self.frame, pady=pady)
        self.frame_body.pack(fill='both',
                             expand=True,
                             )
        # self.frame_body.propagate(0)

        # self.frame_load = tk.Frame(self.frame_body, pady=pady)
        # self.frame_load.pack(side='top', fill='x', expand=True)
        # self._create_frame_load()

        frame_body_left = tk.Frame(self.frame_body, pady=pady)
        frame_body_left.pack(side='top', fill='x', expand=True)
        #
        self.frame_prompt = tk.Frame(frame_body_left, pady=pady)
        self.frame_prompt.pack(side='top', fill='x', expand=True)
        self._create_frame_prompt()

        frame_body_right = tk.Frame(self.frame_body)
        frame_body_right.pack(side='top', fill='x', expand=True)
        #
        self.frame_question = tk.Frame(frame_body_right, pady=pady)
        self.frame_question.pack(side='top', fill='x', expand=True)
        self._create_frame_question()
        #
        self.frame_session_history = tk.Frame(frame_body_right, pady=pady)
        self.frame_session_history.pack(side='top', fill='x', expand=True)
        self._create_frame_session()

    def _create_frame_load(self):
        # frm_load = tk.Frame(self.frame_load)
        # frm_load.pack(side='top')
        frm_load_sx = tk.Frame(self.frame_load)
        frm_load_sx.pack(side='right')
        # button for show/hidden option menu
        tk.Button(self.frame_load,  # self.frame,
                  textvariable=self.option_text_str,
                  anchor='center',
                  command=self._showhidden_option_menu).pack(side='left', expand=False)
        tk.Button(frm_load_sx,
                  text='Load Answers Data',
                  command=self.load_answers_data,
                  anchor='center',
                  ).pack(side='left')
        tk.Button(frm_load_sx,
                  text='Create New Answer Data',
                  command=self._create_new_answers_data,
                  anchor='center',
                  ).pack(side='left')
        tk.Button(frm_load_sx,
                  text='Update Answer Data',
                  command=self.update_answers_data,
                  anchor='center',
                  ).pack(side='left')

    def _create_frame_prompt(self):
        frm_prompt_combo = tk.Frame(self.frame_prompt)
        frm_prompt_combo.pack(side='top',
                              fill='x',
                              expand=False)
        tk.Label(frm_prompt_combo,
                 text='Select Prompt:',
                 anchor='w').pack(side='left', expand=False)
        self.prompts = ttk.Combobox(frm_prompt_combo,
                                    # width=35,
                                    textvariable=self.prompt_name_str,
                                    )
        self.prompts.pack(side='left',
                          fill='x',
                          expand=True)
        tk.Button(frm_prompt_combo,
                  text='comment prompt',
                  command=self.ShowCommentPrompt,
                  anchor='center',
                  ).pack(side='left', fill='x')
        frm_retrieved = tk.Frame(self.frame_prompt)
        frm_retrieved.pack(expand=True,
                           fill='both')

        self.prompt_retrieved = tk.Text(frm_retrieved,
                                        height=15,
                                        width=80,
                                        wrap='word',
                                        )
        self.prompt_retrieved.pack(
            side='left',
            fill='both',
            expand=True)
        self.prompt_retrieved.bind('<FocusOut>',
                                    # '<<Modified>>',
                                   # '<KeyRelease>',
                                   # '<Key>',
                                    self._add_edit_prompt)

        ys = ttk.Scrollbar(frm_retrieved,
                           orient='vertical',
                           command=self.prompt_retrieved.yview)
        self.prompt_retrieved['yscrollcommand'] = ys.set
        ys.pack(side='right', fill='y')

        xs = ttk.Scrollbar(self.frame_prompt,
                           orient='horizontal',
                           command=self.prompt_retrieved.xview)
        self.prompt_retrieved['xscrollcommand'] = xs.set
        xs.pack(side='bottom', fill='x')

    def _create_frame_question(self):
        frm_questions_list_list = tk.Frame(self.frame_question, pady=self.PADY)
        frm_questions_list_list.pack(side='top', fill='x')
        tk.Radiobutton(frm_questions_list_list,
                       text='Ask a list of questions or\nparametrized question\-s',
                       anchor='w',
                       width=21,
                       justify='left',
                       value=self.ASK_QUESTIONS_LIST,
                       variable=self.question_type_of_questions,
                       ).pack(side='left', fill='x')
        self.questions_list = ttk.Combobox(frm_questions_list_list,
                                           width=35,
                                           textvariable=self.questions_list_name)
        self.questions_list.pack(side='left',
                                 fill='x',
                                 expand=True)
        self.questions_list.bind("<<ComboboxSelected>>", self._set_list_question)
        tk.Button(frm_questions_list_list,
                  text='edit questions',
                  command=self.ShowEditQuestionsList).pack(side='left')  # , fill='x', expand=True)
        # tk.Button(frm_questions_list_list,
        #           text='duplicate',
        #           command=self._duplicate_questions_list).pack(side='left', fill='x', expand=True)
        # tk.Button(frm_questions_list_list,
        #           text='delete',
        #           command=self._delete_questions_list).pack(side='left', fill='x', expand=True)

        frm_question_to_ask_single = tk.Frame(self.frame_question, pady=self.PADY)
        frm_question_to_ask_single.pack(side='top', fill='x')
        ttk.Separator(frm_question_to_ask_single).pack(side='top', fill='x', pady=self.PADY)
        tk.Radiobutton(frm_question_to_ask_single,
                       text='Ask a Questions:',
                       anchor='w',
                       width=15,
                       value=self.ASK_SINGLE_QUESTION,
                       variable=self.question_type_of_questions).pack(side='left', ipady=self.PADY, pady=self.PADY)
        self.question_to_ask = tk.Text(frm_question_to_ask_single,
                                       width=55,
                                       height=5,
                                       wrap='word',
                                       insertbackground="white",
                                       # textvariable=self.question_to_ask_string
                                       )
        self.question_to_ask.bind('<Key>', self._set_single_question)
        self.question_to_ask.bind('<Double-Button-1>', self._show_ask_single_question)

        # self.question_to_ask = tk.Entry(frm_question_to_ask_single,
        #                                  width=55,
        #                                  textvariable=self.question_to_ask_string)
        ys = ttk.Scrollbar(frm_question_to_ask_single,
                           orient='vertical',
                           command=self.question_to_ask.yview)
        ys.pack(side='right', fill='y')
        self.question_to_ask.config({"background": "Blue", "foreground": 'White'})
        self.question_to_ask.pack(side='right',
                                  fill='x',
                                  expand=True)
        self.question_to_ask['yscrollcommand'] = ys.set

        tk.Button(self.frame_question,
                  text='Ask Question/-s',
                  command=self._ask_question,
                  anchor='center',
                  ).pack(side='left', fill='x', expand=True)

        frm_GPT_status = tk.Frame(self.frame_question)
        frm_GPT_status.pack(side='top', fill='x')
        tk.Label(frm_GPT_status,
                 text='GPT status:',
                 anchor='w').pack(side='left')
        self.trafic_light = tk.Label(frm_GPT_status,
                                     textvariable=self.gpt_status,
                                     anchor='w')
        self.trafic_light.pack(side='right')

        # tk.Button(self.frame_question,
        #           text='show composed prompt',
        #           command=self._show_composed_prompt).pack(side='left')
        # tk.Button(self.frame_question,
        #           text='Show Answers',
        #           command=self._show_answers_data).pack(side='left')

    def _create_frame_session(self):
        frm_answ = tk.Frame(self.frame_session_history)
        frm_answ.pack(side='top',
                      fill='both',
                      expand=True)
        self.session = tk.Listbox(frm_answ)
        self.session.pack(side='left',
                          fill='both',
                          expand=True)
        self.session.bind('<Double-Button-1>', self.ShowASingleAnswer_session)
        ys = ttk.Scrollbar(frm_answ,
                           orient='vertical',
                           command=self.session.yview)
        self.session['yscrollcommand'] = ys.set
        ys.pack(side='right', fill='y')

        xs = ttk.Scrollbar(self.frame_session_history,
                           orient='horizontal',
                           command=self.session.xview)
        self.session['xscrollcommand'] = xs.set
        xs.pack(side='bottom', fill='x')

    def _set_title(self, filename):
        self.parent.title('Prompt Designer - {}'.format(
            str(Path(filename).name))
        )

    def _load_default_setting(self, *event):
        def ask_what_to_do_interface():
            def create_file(*event):
                loc.destroy()
                self._create_new_answers_data()
                if not self._load_api_keys():
                    messagebox.showwarning('',
                                           'No Api Key found, you must set one')
                    self.ShowHandleApiKeys()

                messagebox.showinfo('',
                                    'No setting configuration found.You must set one.')
                self.ShowParametersSettings()


                # loc.destroy()

            def load_file(*event):
                loc.destroy()
                self.load_answers_data()

                self._load_prompt_names()
                self._load_questions_list()
                if not self._load_api_keys():
                    self.ShowHandleApiKeys()
                # loc.destroy()

            loc = tk.Toplevel(self.parent)
            loc.title('No answer dataset set found')
            tk.Label(loc,
                     text='No answer dataset set found, what do you want to do?\n',
                     justify='center',
                     anchor='center',
                     padx=self.PADX,
                     pady=self.PADY,
                     ).pack(side='top', fill='x')
            frm_ = tk.Frame(loc,
                            padx=self.PADX,
                            pady=self.PADY*3,
                            )
            frm_.pack(side='top', fill='both')
            tk.Button(frm_,
                      text='Create a new file',
                      anchor='center',
                      command=create_file,
                      padx=self.PADX,
                      pady=self.PADY,
                      ).pack(side='left')
            tk.Button(frm_,
                      text='Load file',
                      anchor='center',
                      command=load_file,
                      padx=self.PADX,
                      pady=self.PADY,
                      ).pack(side='right')
            loc.mainloop()

        try:
            settings = LoadJsonData(self.SETTINGS_FILENAME)
        except:
            ask_what_to_do_interface()
            return

        if settings:
            #  load answer data
            if self.data.LoadData(settings[self.SETTING_ANSWERDATA_FILENAME]):
                self._set_title(settings[self.SETTING_ANSWERDATA_FILENAME])
            else:
                ask_what_to_do_interface()
            #  prompt name
            self.prompt_name_str.set(settings[self.SETTING_PROMPT_NAME])
            #  question type
            self.question_type_of_questions.set(settings[self.SETTING_QUESTION_TYPE])
            #  question list name
            self.questions_list_name.set(settings[self.SETTING_QUESTION_LIST])
            # api key
            if not  self._load_api_keys():
                self.ShowHandleApiKeys()
            else:
                self.api_key.set(settings[self.SETTING_API_KEY])
            # self._set_api_key()
            # temperature sampling
            if settings[self.data.TEMPERATURE]:
                self.temperature_sampling_status.set(True)
                self.temperature_str.set(settings[self.data.TEMPERATURE])
            else:
                self.temperature_sampling_status.set(True)
                self.temperature_str.set(0.0)

            if settings[self.data.NUCLEUS]:
                self.nucleus_sampling_status.set(True)
                self.nucleus_str.set(settings[self.data.NUCLEUS])
            else:
                self.nucleus_sampling_status.set(False)

            if settings[self.data.PRESENCE_PENALTY]:
                self.presence_penalty_status.set(True)
                self.presence_penalty_str.set(settings[self.data.PRESENCE_PENALTY])
            else:
                self.presence_penalty_status.set(False)

            if settings[self.data.FREQUENCY_PENALTY]:
                self.frequency_penalty_status.set(True)
                self.frequency_penalty_str.set(settings[self.data.FREQUENCY_PENALTY])
            else:
                self.frequency_penalty_status.set(False)
            self.stop_list = settings[self.data.STOPWORDS_LIST]
            self.n_str.set(settings[self.data.N_])
            self.engine_str.set(settings[self.data.ENGINE])
            self.max_tokens.set(settings[self.data.MAX_TOKEN])

            self._load_session()
        else:
            ask_what_to_do_interface()

            #
            # self._load_prompt_names()
            # self._load_questions_list()

    def _load_session(self):
        self.session.delete(0, 'end')
        session_data = self.data.GetSession()
        for dat_ in session_data:
            # answers = self._extract_answers(dat_)
            self.session.insert('end', dat_)

    def export_gpt3_settings(self, *event):
        filename = Path(filedialog.asksaveasfilename(initialdir = os.getcwd(),
                                                     title = "Please select a file",
                                                     filetypes = (('json file', '*.json'),),
                                                     defaultextension='.json',
                                                     )
                        )
        if filename.name:
            settings = self._get_gtp_parameters()
            settings[self.SETTING_API_KEY] = self.api_keys[self.api_key.get()]
            #  remove unused keys
            for k in ['answers', 'datetime', 'prompt-name', 'prompt-text', 'question', 'response-json']:
                settings.pop(k)
            if SaveJsonData(data=settings,
                            filename=filename.absolute()):
                messagebox.showinfo('',
                                    'Settings Exported')

    def _set_as_default_setting(self, *event):
        settings = self._get_gtp_parameters()
        settings[self.SETTING_API_KEY] = self.api_key.get()
        settings[self.SETTING_ANSWERDATA_FILENAME] = str(self.data.filename)
        settings[self.SETTING_PROMPT_NAME] = self.prompt_name_str.get()
        settings[self.SETTING_QUESTION_TYPE] = self.question_type_of_questions.get()
        settings[self.SETTING_QUESTION_LIST] = self.questions_list_name.get()

        if SaveJsonData(data=settings,
                        filename=self.SETTINGS_FILENAME):
            messagebox.showinfo('',
                                'default settings updated')
        #  close setting frame
        # self.toplevel_settings.destroy()

    def ShowParametersSettings(self, *event):
        def on_close():
            try:
                self.toplevel_settings.destroy()
            except:
                pass
            self.toplevel_settings = None

        if self.toplevel_settings:
            # self.toplevel_settings.grab_set()
            if self.toplevel_settings.winfo_exists():
                self.toplevel_settings.lift
                self.toplevel_settings.focus_set()
                return

        self.toplevel_settings = tk.Toplevel(self.parent)

        self.toplevel_settings.geometry('235x385')
        self.toplevel_settings.title('Parameters')

        self._frame_parameters_settings = tk.Frame(self.toplevel_settings,
                                                   # width=150,
                                                   )

        self._frame_parameters_settings.pack(fill='y',
                                             expand=False)
        tk.Button(self._frame_parameters_settings,  # self.frame,
                  text='close setting',
                  anchor='center',
                  command=on_close, #self.toplevel_settings.destroy
                  ).pack(side='top', fill='x', expand=False)
        ttk.Separator(self._frame_parameters_settings).pack(side='top')
        tk.Button(self._frame_parameters_settings,  # self.frame,
                  text='set as default setting',
                  anchor='center',
                  command=self._set_as_default_setting).pack(side='top', fill='x', expand=False)
        ttk.Separator(self._frame_parameters_settings).pack(side='top')
        tk.Button(self._frame_parameters_settings,  # self.frame,
                  text='Export setting',
                  anchor='center',
                  command=self.export_gpt3_settings).pack(side='top', fill='x', expand=False)

        ttk.Separator(self._frame_parameters_settings).pack(side='top')
        ttk.Separator(self._frame_parameters_settings).pack(side='top', fill='x')

        frm_api_key = tk.Frame(self._frame_parameters_settings)
        frm_api_key.pack(side='top')
        # ttk.Separator(self._frame_parameters_settings).pack(side='top', fill='x')
        ttk.Checkbutton(self._frame_parameters_settings,
                        text='Record Response Json',
                        variable=self.record_json_status).pack(side='top', fill='x')
        ttk.Separator(self._frame_parameters_settings).pack(side='top', fill='x')

        tk.Label(frm_api_key,
                 text='select API key:',
                 ).pack(side='left')
        self.keys_api = ttk.Combobox(frm_api_key,
                                     width=15,
                                     textvariable=self.api_key)
        self.keys_api.pack(side='left')
        self.keys_api['values'] = sorted(list(self.api_keys.keys()))

        frm_setting_options = tk.Frame(self._frame_parameters_settings)
        frm_setting_options.pack(side='top', fill='x')

        frm_settings_temperature_sampling = tk.Frame(frm_setting_options)
        frm_settings_temperature_sampling.pack(side='top')
        tk.Checkbutton(frm_settings_temperature_sampling,
                       text='temperature sampling',
                       # value=self.SAMPLING_TEMPERATURE,
                       anchor='w',
                       width=18,
                       variable=self.temperature_sampling_status).pack(side='left')

        self.temperatures = ttk.Combobox(frm_settings_temperature_sampling,
                                         width=5,
                                         textvariable=self.temperature_str)
        self.temperatures.pack(side='right',
                               # fill='x',
                               expand=False)
        ToolTipParameters(frm_settings_temperature_sampling, _temperature_message)

        frm_settings_nucleus_sampling = tk.Frame(frm_setting_options)
        frm_settings_nucleus_sampling.pack(side='top')
        tk.Checkbutton(frm_settings_nucleus_sampling,
                       text='nucleus sampling',
                       # value=self.SAMPLING_NUCLEUS,
                       anchor='w',
                       width=18,
                       variable=self.nucleus_sampling_status).pack(side='left')
        # tk.Label(frm_cmbopt, text='select nucleus top_p:').pack(side='left')
        self.nucleus = ttk.Combobox(frm_settings_nucleus_sampling,
                                    width=5,
                                    textvariable=self.nucleus_str)
        self.nucleus.pack(side='right',
                          # fill='x',
                          expand=False)
        ToolTipParameters(frm_settings_nucleus_sampling, _nucleus_message)

        frm_settings_engine = tk.Frame(self._frame_parameters_settings)
        frm_settings_engine.pack(side='top')
        tk.Label(frm_settings_engine,
                 text='Select engine',
                 width=11,
                 anchor='w').pack(side='left')
        self.engine = ttk.Combobox(frm_settings_engine,
                                   widt=17,
                                   textvariable=self.engine_str)
        self.engine.pack(side='right')
        ToolTipParameters(frm_settings_engine, _engine_message)

        tk.Checkbutton(self._frame_parameters_settings,
                       text='avoid asking the same question',
                       variable=self._avoid_asking_same_question,
                       anchor='w',
                       ).pack(side='top', fill='x')

        frm_n_ = tk.Frame(self._frame_parameters_settings)
        frm_n_.pack(side='top', fill='x')
        tk.Label(frm_n_,
                 text='select n_').pack(side='left')
        self.n_ = ttk.Combobox(frm_n_,
                               width=4,
                               textvariable=self.n_str)
        self.n_.pack(side='right', fill='x')

        frm_stop_list = tk.Frame(self._frame_parameters_settings)
        frm_stop_list.pack(side='top', fill='x')

        frm_stop_list_l = tk.Frame(frm_stop_list)
        frm_stop_list_l.pack(side='top', fill='x')
        tk.Label(frm_stop_list_l,
                 anchor='w',
                 text='Stopwords').pack(side='left')
        tk.Entry(frm_stop_list_l,
                 width=10,
                 textvariable=self.stopword,
                 ).pack(side='left')
        tk.Button(frm_stop_list_l,
                  text='Add',
                  command=self._add_stopword_to_stopwords_list,
                  ).pack(side='left')
        frm_stop_list_ = tk.Frame(frm_stop_list)
        frm_stop_list_.pack(side='right', fill='x')
        tk.Button(frm_stop_list_,
                  text='load default',
                  command=self._load_default_stop_list).pack(side='left')
        self.lst_stop_list = tk.Listbox(frm_stop_list_,
                                        width=11,
                                        height=3)
        self.lst_stop_list.pack(side='left')
        ys = ttk.Scrollbar(frm_stop_list_, orient='vertical', command=self.lst_stop_list.yview)
        self.lst_stop_list['yscrollcommand'] = ys.set
        ys.pack(side='right', fill='y')
        self.lst_stop_list.bind('<Double-Button-1>', self._delete_stopword_from_list)

        frm_max_token = tk.Frame(self._frame_parameters_settings)
        frm_max_token.pack(side='top', fill='x')
        #
        tk.Label(frm_max_token,
                 text='Max Tokens: ').pack(side='left')
        tk.Entry(frm_max_token,
                 textvariable=self.max_tokens,
                 width=5
                 ).pack(side='right')
        # self.max_tokens.set(150)
        frm_penalties = tk.Frame(self._frame_parameters_settings)
        frm_penalties.pack(side='top', fill='x')
        #
        frm_presence_penalty = tk.Frame(frm_penalties)
        frm_presence_penalty.pack(side='top', fill='x')
        tk.Checkbutton(frm_presence_penalty,
                       text='presence_penalty',
                       variable=self.presence_penalty_status).pack(side='left')
        self.presence_penalty = ttk.Combobox(frm_presence_penalty,
                                             textvariable=self.presence_penalty_str)
        self.presence_penalty.pack(side='right')
        self.presence_penalty['values'] = [float(x / 100) for x in range(-100, 201, 20)]
        # self.presence_penalty_str.set(self.data.t(0.))
        ToolTipParameters(frm_presence_penalty, _presence_penalty_message)

        frm_frequency_penalty = tk.Frame(frm_penalties)
        frm_frequency_penalty.pack(side='top', fill='x')
        tk.Checkbutton(frm_frequency_penalty,
                       text='frequency_penalty',
                       variable=self.frequency_penalty_status).pack(side='left')
        self.frequency_penalty = ttk.Combobox(frm_frequency_penalty,
                                              textvariable=self.frequency_penalty_str)
        self.frequency_penalty['values'] = [float(x / 100) for x in range(-100, 201, 20)]
        # self.frequency_penalty_str.set(self.data.t(0.))
        self.frequency_penalty.pack(side='right')
        ToolTipParameters(frm_frequency_penalty, _frequency_penalty_message)
        # self._load_api_keys()
        self._load_temperatures_list()
        self._load_nucleus_list()
        self._set_gpt_status_ready()
        self._load_engines_list()
        self._load_n_()
        self._load_stop_list()

        self.toplevel_settings.mainloop()

    def _add_stopword_to_stopwords_list(self, *event):
        if self.stopword.get():
            sw = self.stopword.get()
        else:
            sw = '\n'
        self.lst_stop_list.insert(-1, sw)
        self.stop_list.append(sw)

    def _load_engines_list(self):
        self.engine['values'] = self.ENGINES_LIST
        # self.engine_str.set(self.engine['values'][0])

    def _load_nucleus_list(self):
        self.nucleus['values'] = sorted([self.data.t(t)
                                         for t in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]])
        # self.nucleus_str.set(self.nucleus['values'][0])

    def _load_temperatures_list(self):
        self.temperatures['values'] = sorted([self.data.t(t)
                                              for t in [0, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2]])
        # self.temperature_str.set(self.temperatures['values'][0])

    def _compose_answer_pattern(self,
                                header,
                                body,
                                foot):
        return '\n'.join([foot,
                          header,
                          body,
                          foot])

    def _load_prompt_names(self, *event):
        prompt_names = self.data.GetPromptNames()
        if len(prompt_names) > 0:
            self.prompts['values'] = sorted(self.data.GetPromptNames())
            self.prompt_name_str.set(self.prompts['values'][0])
        else:
            self.prompts['values'] = list()
            self.prompt_name_str.set('default-prompt-name')

    def _load_stop_list(self):
        try:
            stoplist = self.stop_list
        except:
            stoplist = self.DEFAULT_STOPWORDS
        stoplist = sorted(list(set(stoplist)))
        self.lst_stop_list.delete(0, 'end')
        for sw in stoplist:  # self.DEFAULT_STOPWORDS:
            self.lst_stop_list.insert(0, sw)

    def _load_default_stop_list(self):
        self.lst_stop_list.delete(0, 'end')
        for sw in self.DEFAULT_STOPWORDS:
            self.lst_stop_list.insert(0, sw)
        self.stop_list = self.DEFAULT_STOPWORDS.copy()

    def _load_n_(self, *event):
        self.n_['values'] = [1, 2, 3, 5, 10]
        self.n_str.set(1)

    def _get_gtp_parameters(self):
        def fix_type(parameter):
            try:
                parameter = self.data.t(parameter)
            except:
                # 'all'
                pass
            return parameter

        parameters = self.data.GetAnswerDataEmptyDict()
        parameters[self.data.PROMPT] = self.prompt_name_str.get()
        parameters[self.data.N_] = self.n_str.get()
        parameters[self.data.ENGINE] = self.engine_str.get()
        parameters[self.data.DATETIME] = localtime()
        # parameters[self.data.SAMPLING_MODE] = self.sampling_technique.get()

        if self.temperature_sampling_status.get():
            parameters[self.data.TEMPERATURE] = fix_type(
                self.temperature_str.get())  # self.data.t(self.temperature_str.get())
        if self.nucleus_sampling_status.get():
            parameters[self.data.NUCLEUS] = fix_type(self.nucleus_str.get())  # self.data.t(self.nucleus_str.get())
        if self.frequency_penalty_status.get():
            parameters[self.data.FREQUENCY_PENALTY] = fix_type(
                self.frequency_penalty_str.get())  # self.data.t(self.frequency_penalty_str.get())
        if self.presence_penalty_status.get():
            parameters[self.data.PRESENCE_PENALTY] = fix_type(
                self.presence_penalty_str.get())  # self.data.t(self.presence_penalty_str.get())
        parameters[self.data.MAX_TOKEN] = self.max_tokens.get()
        try:
            parameters[self.data.STOPWORDS_LIST] = self.stop_list
        except AttributeError: #  AttributeError: 'PromptDesigner' object has no attribute 'stop_list'
            self.stop_list = self.lst_stop_list.get(0, 'end')

            parameters[self.data.STOPWORDS_LIST] = self.stop_list

        return parameters


    def _delete_stopword_from_list(self, *event):
        if self.lst_stop_list.curselection():
            item_to_delete = self.lst_stop_list.get(self.lst_stop_list.curselection())
            if item_to_delete == '':
                item_to_delete = '\n'
            self.lst_stop_list.delete(self.lst_stop_list.curselection())
            # workaround, it should be fixed in init
    # def _get_stopwordslist(self):
    #     return self.stop_list

    def _compose_prompt(self,
                        question='',
                        prompt_template=''):
        # try:
        composed = prompt_template.format(question=question)
        # except KeyError:
        #  if no key question found
        if composed == prompt_template:
            composed = prompt_template + question
        return composed

    def _ask_single_question(self,
                             parameters):
        self._set_gpt_status_asking()
        sleep(0.9)

        #  check if avoid asking the same question is True
        if self._avoid_asking_same_question.get() and \
                self.data.IsQuestionAsked(question=parameters[self.data.QUESTION],
                                          filters=parameters):
            messagebox.showinfo('',
                                'You have already asked "{}"'.format(parameters[self.data.QUESTION]))
            return

        try:
            response = self.AskGPT3(parameters)
            self._set_gpt_status_ready()
        except:
            # whatever openAi error occurs
            self._set_gpt_status_not_ready()
            return
        #  extract text
        answers = self._extract_answers(response)
        #  record answers
        if self.record_json_status.get():
            parameters[self.data.JSON_RESPONSE] = response
        #  add answer/-s
        self.data.AddQuestionAnswer(answers=answers,
                                    parameters=parameters)
        #  show answers
        self.update_session_with_answer(answers=answers,
                                        parameters=parameters)

    def _extract_answers(self, answers):
        return [ans['text'] for ans in answers['choices']]

    def get_questions_list(self):
        def get_line():
            for line in self.data.GetQuestionsList(
                    self.questions_list_name.get()).split('\n'):
                yield line

        questions_list = list()
        recording_question_block = False
        question_block = list()

        for line in get_line():
            line = line.strip()
            if not line:
                continue
            if line == self.QUESTION_BLOCK:
                if recording_question_block:
                    # stop recording
                    question = '\n'.join(question_block).strip()
                    questions_list.append(question)

                    question_block = list()
                    recording_question_block = False
                else:
                    recording_question_block = True
            elif line.startswith(self.COMMENT_BLOCK):
                continue
            else:
                if recording_question_block:
                    question_block.append(line)
                else:
                    questions_list.append(line)
        if question_block:
            question = '\n'.join(question_block).strip()
            questions_list.append(question)
        return questions_list

    def ask_parametrized_questions_list(self,
                                        parameters):
        ql = self.data.GetQuestionsList(self.questions_list_name.get())
        ql_configurations = self.get_parametrized_questions_list_parameters_items(ql)
        for configuration in ql_configurations:
            parameters_question = parameters.copy()
            configuration_txt = ' | '.join(['{}: {}'.format(k, v)
                                           for k, v in configuration.items()]).strip()
            parameters_question[self.data.QUESTION] = configuration_txt
            prompt_text = self._compose_prompt_for_question_with_parameters(configuration=configuration,
                                                                            prompt_template=self.prompt_retrieved.get(
                                                                                '1.0', 'end-1c'))
            parameters_question[self.data.PROMPT_TEXT] = prompt_text
            self._ask_single_question(parameters=parameters_question)

    def _ask_questions_list(self,
                            parameters):
        ql = self.data.GetQuestionsList(self.questions_list_name.get())
        if self.__is_a_parametrized_question(ql):
            self.ask_parametrized_questions_list(parameters)
            return

        for question in self.get_questions_list():
            parameters_question = deepcopy(parameters)
            parameters_question[self.data.QUESTION] = question
            prompt_text = self._compose_prompt(question=question,
                                                   prompt_template=self.prompt_retrieved.get('1.0', 'end-1c'))
            parameters_question[self.data.PROMPT_TEXT] = prompt_text
            self._ask_single_question(parameters=parameters_question)

    def _ask_question(self, *event):
        # sleep(0.5)
        # self._set_gpt_status_asking()
        #
        parameters = self._get_gtp_parameters()
        #  ask
        if self.question_type_of_questions.get() == self.ASK_SINGLE_QUESTION:
            parameters[self.data.QUESTION] = self.question_to_ask.get('1.0',
                                                                      'end-1c')  # self.question_to_ask_string.get()

            prompt_text = self._compose_prompt(question=parameters[self.data.QUESTION],
                                               # self.question_to_ask_string.get(),
                                               prompt_template=self.prompt_retrieved.get('1.0', 'end-1c'))
            parameters[self.data.PROMPT_TEXT] = prompt_text

            self._ask_single_question(parameters)
        else:
            self._ask_questions_list(parameters)

        #  update answers data
        self.data.UpdateData()

    def __is_a_parametrized_question(self,
                                     question: str):
        if self.get_parametrized_questions_list_parameters_items(question):
            return True
        return False

    def _compose_prompt_for_question_with_parameters(self,
                                                     configuration: dict,
                                                     prompt_template: str)-> str:
        try:
            return prompt_template.format(**configuration)
        except Exception as err:
            messagebox.showerror('',
                                 str(err))


    def update_session_with_answer(self,
                                   answers: list,
                                   parameters: dict):

        for answer in answers:
            ans = '\tA: {}'.format(answer.strip())
            self.session.insert(0, ans)
        quest = 'Q: {}'.format(parameters[self.data.QUESTION])
        self.session.insert(0, quest)
        self.selection_clear()
        self.session.select_set(0)

        session_data = self.session.get(0, 'end')
        self.data.UpdateSession(session=session_data)
        self.update_answers_data()

    def load_answers_data(self, *event):
        filename = Path(filedialog.askopenfilename(initialdir=os.getcwd(),
                                                   title="Please select a file",
                                                   filetypes=(('all files', '*.*'),
                                                              ('Prompt designer file', '*.answer-data'),
                                                              ),
                                                   defaultextension='.answer-data'
                                                   )
                        )
        if filename.name:
            if self.data.LoadData(str(filename.absolute())):
                messagebox.showinfo('',
                                    'answers data loaded')

                self._set_title(str(filename))

                self._load_prompt_names()
                self._load_questions_list()

                self._load_session()


    def _create_new_answers_data(self, *event):
        filename = Path(filedialog.asksaveasfilename(initialdir=os.getcwd(),
                                                   title="Please select a file",
                                                   filetypes=(('all files', '*.*'),
                                                              ('Prompt designer file', '*.answer-data'),
                                                              ),
                                                   defaultextension='.answer-data'
                                                   )
                        )
        if filename.name:
            self.data = PromptDesignerDataset(filename=str(filename.absolute()))
            self.data.UpdateData()
            self._set_title(str(filename))
            self._load_prompt_names()
            self._load_questions_list()

            self._clear_session()


    def save_as_answers_data(self, *event):
        filename = Path(filedialog.asksaveasfilename(initialdir=os.getcwd(),
                                                   title="Please select a file",
                                                   filetypes=(('all files', '*.*'),
                                                              ('Prompt designer file', '*.answer-data'),
                                                              ),
                                                   defaultextension='.answer-data'
                                                   )
                        )
        if filename.name:
            self.data.filename = str(filename.absolute())
            self._set_title(self.data.filename)
        if not self.data.UpdateData():
            messagebox.showerror('',
                                 'Unknown Error. Please re-start Prompt Designer')

    def update_answers_data(self, *event):
        if not self.data.filename:
            filename = Path(filedialog.asksaveasfilename(initialdir=os.getcwd(),
                                                           title="Please select a file",
                                                           filetypes=(('all files', '*.*'),
                                                                      ('Prompt designer file', '*.answer-data'),
                                                                      ),
                                                           defaultextension='.answer-data'
                                                           )
                            )
            if filename.name:
                self.data.filename = str(filename.absolute())
                self._set_title(self.data.filename)
        if not self.data.UpdateData():
            messagebox.showerror('',
                                 'Unknown Error. Please re-start Prompt Designer')
            # return
            # messagebox.showinfo('',
            #                     'answers data updated')

    def _add_edit_prompt(self, *event):

        prompt_name = self.prompt_name_str.get()
        if prompt_name:
            self.data.AddPrompt(prompt_name,
                                self.prompt_retrieved.get("1.0", 'end-1c'))
        if type(self.prompts['values']) in [list, tuple]:
            if not prompt_name in self.prompts['values']:  # self.data.IsPromptPresent(self.prompt_name_str.get()):
                vals = list(self.prompts['values'])
                vals.append(prompt_name)
                self.prompts['values'] = vals
                # except AttributeError:
                #     self.prompts['values'] = list([prompt_name])
        else:
            self.prompts['values'] = list([prompt_name])
        self.update_answers_data()

    def _delete_questions_list(self, *event):
        message = 'Are you sure you want to delete |{}| question list?'.format(self.questions_list_name.get())
        answer = messagebox.askquestion('',
                                        message)
        if answer == 'yes':
            self.data.RemoveQuestionsList(self.questions_list_name.get())
            self._load_questions_list()
            self.data.UpdateData()
        self.update_answers_data()

    def _show_answers_data(self, *event):
        ans = tk.Toplevel(self.parent)
        answersdatainterface = AnswersInterface(ans,
                                                self.data)
        # ans.title('Answers')
        ans.mainloop()

    def _export_session(self, *event):
        filename = Path(filedialog.asksaveasfilename(initialdir=self.default_path,
                                                   title="Select file",
                                                   filetypes=self.FILE_TYPE_SESSION,))
        if filename.name:
            self.data.ExportSession(str(filename.absolute()))
            messagebox.showinfo('',
                                'Session exported')

    def _export_answer_data(self, *event):
        exp = tk.Toplevel(self.parent)
        exp_interface = ExportSessionData(exp,
                                          self.data)
        exp.mainloop()

    def _show_ask_single_question(self, *event):
        def on_exit(*event):
            self.question_to_ask.delete('1.0', 'end')
            self.question_to_ask.insert('1.0', ql_txt.get('1.0', 'end-1c'))
            ql.destroy()
        # self._shared_ask_single_question_str.set(self.question_to_ask.get('1.0', 'end-1c'))
        #  lunch window
        ql = tk.Toplevel(self.parent)
        ql.title('Question to ask')
        ql_txt = tk.Text(ql,
                         bg='blue',
                         fg='white',
                         wrap='word',
                         insertbackground='white')
        ql_txt.configure(font=("Times New Roman", 15))
        ql_txt.pack(fill='both', expand=True)
        ql_txt.insert('end', self.question_to_ask.get('1.0', 'end-1c'))
        ql_txt.focus_set()
        tk.Button(ql,
                  text='Save',
                  command=on_exit).pack(side='bottom', fill='x')
        ql.mainloop()

    def _show_composed_prompt(self, *event):
        ql = tk.Toplevel(self.parent)
        ql.title('Prompt be submitted')
        ql_txt = tk.Text(ql,
                         bg='blue',
                         fg='white',
                         insertbackground="white",
                         wrap='word',
                         padx=self.PADX,
                         pady=self.PADY)
        ql_txt.configure(font=("Times New Roman", 15))
        ql_txt.pack(fill='both', expand=True)
        ql_txt.insert('end', self._compose_prompt(question=self.question_to_ask.get('1.0', 'end-1c'),
                                                  # self.question_to_ask_string.get(),
                                                  prompt_template=self.prompt_retrieved.get("1.0", 'end-1c')))
        ql_txt.focus_set()
        tk.Button(ql,
                  text='Close',
                  command=lambda: ql.destroy()).pack(side='bottom', fill='x')
        ql.mainloop()

    def ShowEditQuestionsList(self, *event):
        def on_exit(*event):
            try:
                self._shared_questions_list_str.set(ql_txt.get('1.0', 'end-1c'))
                self.toplevel_ql.destroy()
                self.data.AddQuestionsList(self.questions_list_name.get(),
                                           self._shared_questions_list_str.get())
                if self.questions_list_name.get() not in self.questions_list[
                    'values']:  # self.data.IsQuestionsListPresent(self.questions_list_name.get()):
                    vals = list(self.questions_list['values'])
                    vals.append(self.questions_list_name.get())
                    self.questions_list['values'] = vals
                self._set_list_question()
                self.update_answers_data()
            except:
                pass
            self.toplevel_ql = None

        if self.toplevel_ql:
            # self.toplevel_settings.grab_set()
            if self.toplevel_ql.winfo_exists():
                self.toplevel_ql.lift
                self.toplevel_ql.focus_set()
                return

        # if the name is present, retrieve the list
        if self.data.IsQuestionsListPresent(self.questions_list_name.get()):
            self._shared_questions_list_str.set(self.data.GetQuestionsList(self.questions_list_name.get()))
        # otherwise, start from scratch
        else:
            self._shared_questions_list_str.set(self.EMPTY_QUESTIONS_LIST)
        #  lunch window
        self.toplevel_ql = tk.Toplevel(self.parent)
        self.toplevel_ql.title(self.questions_list_name.get())

        frm_help = tk.Frame(self.toplevel_ql)
        frm_help.pack(side='top', fill='x', expand=True)

        help_ql = tk.Label(frm_help,
                           text='how to write a questions list',
                           relief='ridge',
                           )
        help_ql.pack(side='left', fill='x', expand=True)
        ToolTipQuestionsList(help_ql, message=_help_questions_list)

        help_pq = tk.Label(frm_help,
                          text='how to write a parametrized question/-s',
                          relief='ridge',
                          )
        help_pq.pack(side='left', fill='x', expand=True)
        ToolTipQuestionsList(help_pq, message=_help_parametrized_question)
        help_leg = tk.Label(frm_help,
                            text='Symbols legend',
                            relief='ridge',
                          )
        help_leg.pack(side='left', fill='x', expand=True)
        ToolTipQuestionsList(help_leg, message=_help_legend)

        ql_txt = tk.Text(self.toplevel_ql,
                         bg='blue',
                         fg='white',
                         wrap='word',
                         insertbackground='white')
        ql_txt.configure(font=("Times New Roman", 15))
        ql_txt.pack(fill='both', expand=True)
        ql_txt.insert('end', self._shared_questions_list_str.get())
        ql_txt.focus_set()
        tk.Button(self.toplevel_ql,
                  text='Save',
                  command=on_exit).pack(side='bottom', fill='x')
        self.toplevel_ql.mainloop()


        self.update_answers_data()

    def _load_questions_list(self, *event):
        self.questions_list['values'] = self.data.GetQuestionsListNames()
        try:
            self.questions_list_name.set(self.questions_list['values'][0])
        except IndexError:
            self.questions_list_name.set('questions-list-name')

    def _show_prompt(self, *event):
        prompt = self.data.GetPrompt(self.prompt_name_str.get())
        try:
            self.prompt_retrieved.delete("1.0", 'end')
        except:
            pass
        self.prompt_retrieved.insert('end-1c', prompt)
        self.prompt_retrieved.delete('end-1c')

    def _delete_prompt(self, *event):
        message = 'Are you sure you want to delete |{}| prompt?'.format(self.prompt_name_str.get())
        answer = messagebox.askquestion('',
                                        message)
        if answer == 'yes':
            self.data.RemovePrompt(self.prompt_name_str.get())
            self._load_prompt_names()
            self.data.UpdateData()

    def ShowCommentPrompt(self, *event):

        def on_exit(*event):
            self._shared_comment_prompt_str.set(comment_txt.get('1.0', 'end-1c'))
            comment.destroy()
            self.data.AddCommentPrompt(self.prompt_name_str.get(),
                                       self._shared_comment_prompt_str.get())
            self.update_answers_data()

        # if the name is present, retrieve the list
        if self.data.IsCommentPromptPresent(self.prompt_name_str.get()):
            self._shared_comment_prompt_str.set(self.data.GetCommentPrompt(self.prompt_name_str.get()))
        # otherwise, start from scratch
        else:
            self._shared_comment_prompt_str.set(self.EMPTY_COMMENT)
        #  lunch window
        comment = tk.Toplevel(self.parent)
        comment.title('Comment Prompt')
        comment_txt = tk.Text(comment,
                              bg='blue',
                              fg='white',
                              wrap='word',
                              insertbackground='white')
        comment_txt.configure(font=("Times New Roman", 15))
        comment_txt.pack(fill='both', expand=True)
        comment_txt.insert('end', self._shared_comment_prompt_str.get())
        comment_txt.focus_set()
        tk.Button(comment,
                  text='Save comment',
                  command=on_exit).pack(side='bottom', fill='x')
        comment.mainloop()

    def _duplicate_prompt(self, *event):
        new_prompt_name = simpledialog.askstring('Prompt Name',
                                                 'prompt name: ',
                                                 initialvalue=self.prompt_name_str.get())
        if new_prompt_name:
            self.data.AddPrompt(new_prompt_name, self.data.GetPrompt(self.prompt_name_str.get()))
            self.prompt_name_str.set(new_prompt_name)
            self._add_edit_prompt()
            self.update_answers_data()

    def _export_question_list(self, *event):
        filename = Path(filedialog.askdirectory(initialdir=self.default_path,
                                                title="Select directory",
                                                   # filetypes=self.FILE_TYPE_QUESTIONS_LIST,
                                                ))
        if filename.name:
            filename = str(filename.joinpath(self.questions_list_name.get()+'.questions-list'))
            codecs.open(filename, 'w', 'utf-8').write(
                self.data.GetQuestionsList(self.questions_list_name.get())
            )

            messagebox.showinfo('', 'Questions list exported')

    def _export_question_lists(self, *event):
        filename = Path(filedialog.askdirectory(initialdir=self.default_path,
                                                title="Select directory",
                                                   # filetypes=self.FILE_TYPE_QUESTIONS_LIST,
                                                ))
        if filename.name:
            for question_list in self.data.GetQuestionsListNames():
                fileout = str(filename.joinpath(question_list+'.questions-list'))
                codecs.open(fileout, 'w', 'utf-8').write(
                    self.data.GetQuestionsList(self.questions_list_name.get())
                )
            messagebox.showinfo('', 'Questions lists exported')

    def ExportPdf_QuestionsListAndAnswers(self, *event):
        def fix_chars(text):
            return text.encode('latin-1', 'replace').decode('latin-1')

        color_dark_blue = (0,0,139)
        color_black = (0, 0, 0)
        color_bk = (255, 255, 255)
        color_red = (220,20,60)
        color_blue = (0, 0, 255)
        color_confl_blue = (100,149,237)
        color_blueviolette = (138, 43, 226)

        width = 200 # mm

        dirout = Path(filedialog.askdirectory(initialdir=self.default_path,
                                              title="Select directory",))
        if not dirout.name:
            return
        fileout = str(dirout.joinpath(self.questions_list_name.get()+'.pdf'))

        fpdf.set_global("SYSTEM_TTFONTS", os.path.join(os.path.dirname(__file__), 'fonts'))

        pdf = FPDF()
        pdf.set_margins(1.5, 1, 1.5)
        pdf.add_page()
        pdf.set_author('Prompt Designer')
        pdf.set_title('Q&A session')

        pdf.set_font("Arial", size=15)
        pdf.set_text_color(*color_dark_blue)
        question_title = 'Q&A session - {}'.format(self.questions_list_name.get())

        pdf.multi_cell(width, 10,
                       txt=fix_chars(question_title),
                       align="C")
        # pdf.ln()
        #  check if it is a parametrized questions or not
        ql = self.data.GetQuestionsList(self.questions_list_name.get())
        if self.__is_a_parametrized_question(ql):
            ql_configurations = self.get_parametrized_questions_list_parameters_items(ql)
            for configuration in ql_configurations:
                configuration_txt = ' | '.join(['{}: {}'.format(k, v)
                                                for k, v in configuration.items()]).strip()
                question = configuration_txt

                pdf.set_text_color(*color_black)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(width, 10, txt=fix_chars(question))  # , align="J", )

                # parameters_question = parameters.copy()
                answers = self.data.GetAnswers(question)
            #
            #     parameters_question[self.data.QUESTION] = configuration_txt
            #     prompt_text = self._compose_prompt_for_question_with_parameters(configuration=configuration,
            #                                                                     prompt_template=self.prompt_retrieved.get(
            #                                                                         '1.0', 'end-1c'))
            #     parameters_question[self.data.PROMPT_TEXT] = prompt_text
            #
            #
            # self.get_parametrized_questions_list_parameters_items(ql)
            # for question in self.get_questions_list():
            #     pdf.set_text_color(*color_black)
            #     pdf.set_font("Arial", size=15)
            #     pdf.multi_cell(width, 10, txt=fix_chars(question))  # , align="J", )
            #
            #     answers = self.data.GetAnswers(question)
                for answer in answers:
                    header_answer = self._ANSWER_PATTERN_HEADER.format(engine=answer[self.data.ENGINE],
                                                                       prompt=answer[self.data.PROMPT],
                                                                       temperature=answer[self.data.TEMPERATURE],
                                                                       nucleus=answer[self.data.NUCLEUS],
                                                                       presence_penalty=answer[
                                                                           self.data.PRESENCE_PENALTY],
                                                                       frequency_penalty=answer[
                                                                           self.data.FREQUENCY_PENALTY],
                                                                       )
                    pdf.set_font("Arial", size=8)
                    pdf.set_text_color(*color_red)
                    pdf.multi_cell(width,
                                   5,
                                   txt=fix_chars(header_answer),
                                   # align="J",
                                   )

                    for ans_text in answer[self.data.ANSWERS]:
                        answer_text = '--  A: {}'.format(ans_text)
                        pdf.set_font("Arial", "I", size=12)
                        pdf.set_text_color(*color_confl_blue)
                        pdf.multi_cell(width, 10,
                                       txt=fix_chars(answer_text),
                                       align="J",
                                       )

        else:

            for question in self.get_questions_list():
                pdf.set_text_color(*color_black)
                pdf.set_font("Arial", size=15)
                pdf.multi_cell(width, 10, txt=fix_chars(question)) #, align="J", )

                answers = self.data.GetAnswers(question)
                for answer in answers:
                    header_answer = self._ANSWER_PATTERN_HEADER.format(engine=answer[self.data.ENGINE],
                                                       prompt=answer[self.data.PROMPT],
                                                       temperature=answer[self.data.TEMPERATURE],
                                                       nucleus=answer[self.data.NUCLEUS],
                                                       presence_penalty=answer[self.data.PRESENCE_PENALTY],
                                                       frequency_penalty=answer[self.data.FREQUENCY_PENALTY],
                                                       )
                    pdf.set_font("Arial", size=8)
                    pdf.set_text_color(*color_red)
                    pdf.multi_cell(width,
                                   5,
                                   txt=fix_chars(header_answer),
                                   # align="J",
                                   )

                    for ans_text in answer[self.data.ANSWERS]:
                        answer_text = '--  A: {}'.format(ans_text)
                        pdf.set_font("Arial", "I", size=12)
                        pdf.set_text_color(*color_confl_blue)
                        pdf.multi_cell(width, 10,
                                    txt=fix_chars(answer_text),
                                    align="J",
                                    )

        pdf.output(fileout)
        messagebox.showinfo('',
                            'Q&A pdf exported')

    def _duplicate_questions_list(self, *event):
        new_questionlist_name = simpledialog.askstring('Question list',
                                                       'question list name: ',
                                                       initialvalue=self.questions_list_name.get())
        if new_questionlist_name:
            self.data.AddQuestionsList(question_list_name=new_questionlist_name,
                                       questions=self.data.GetQuestionsList(self.questions_list_name.get()))

            if new_questionlist_name not in self.questions_list[
                'values']:  # self.data.IsQuestionsListPresent(self.questions_list_name.get()):
                vals = list(self.questions_list['values'])
                vals.append(new_questionlist_name)
                self.questions_list['values'] = vals

                self.questions_list_name.set(new_questionlist_name)

            self.update_answers_data()

    def _set_single_question(self, *event):
        self.question_type_of_questions.set(self.ASK_SINGLE_QUESTION)

    def _set_list_question(self, *event):
        self.question_type_of_questions.set(self.ASK_QUESTIONS_LIST)
        try:
            self.toplevel_ql.destroy()
        except:
            pass
        self.toplevel_ql = None

    def import_questions_lists(self, *event):
        directory = Path(filedialog.askdirectory(initialdir=self.default_path,
                                               title="Select prompt file"))
        if directory.name:
            pattern = str(directory.joinpath('*.questions-list'))
            for file in glob(pattern):
                file = Path(file)
                self.__import_a_questions_list(file)
        self._load_questions_list()

    def import_questions_list(self, *event):
        file = Path(filedialog.askopenfilename(initialdir=self.default_path,
                                               title="Select prompt file",
                                               filetypes=self.FILE_TYPE_QUESTIONS_LIST))
        if file.name:
            self.__import_a_questions_list(file)
        self._load_questions_list()

    def __import_a_questions_list(self,
                               filename: Path):
        question_list_name = filename.name.replace(filename.suffix, '')

        question_list = codecs.open(str(filename.absolute()), 'r', 'utf-8').readlines()
        question_list = ''.join(question_list)
        self.data.AddQuestionsList(question_list_name=question_list_name,
                                   questions=question_list)
        self.data.UpdateData()

    def import_prompt(self, *event):
        file = Path(filedialog.askopenfilename(initialdir=self.default_path,
                                               title="Select prompt file",
                                               filetypes=self.FILE_TYPE_PROMPT))
        if file.name:
            self.__import_a_single_prompt(file)
            self.data.UpdateData()
            self._load_prompt_names()

    def import_prompts(self, *event):
        directory = Path(filedialog.askdirectory(initialdir=self.default_path,
                                                 title="Select directory",))

        if not directory.name:
            return
        pattern = directory.joinpath('*.prompt')
        files = glob(str(pattern.absolute()))
        for file in files:
            self.__import_a_single_prompt(file)
        self.data.UpdateData()
        self._load_prompt_names()
        # self._load_questions_list()
        # self._load_session()

    def __import_a_single_prompt(self, prompt_filename):
        record_comment = False
        record_prompt = False

        prompt_name = str(Path(prompt_filename).name).replace('.prompt', '')
        prompt = list()
        prompt_comment = list()

        for line in codecs.open(prompt_filename, 'r', 'utf-8').readlines():
            if line.strip() == self.EXPORT_PROMPT_COMMENT_BEGIN:
                record_comment = True
            elif line.strip() == self.EXPORT_PROMPT_PROMPT_BEGIN:
                record_comment = False
                record_prompt = True
            else:
                if record_comment:
                    prompt_comment.append(line)
                elif record_prompt:
                    prompt.append(line)

        prompt_comment = ''.join(prompt_comment).strip()
        prompt = ''.join(prompt).strip()

        self.data.AddPrompt(prompt_name=prompt_name,
                            prompt=prompt)
        self.data.AddCommentPrompt(prompt_name=prompt_name,
                                   comment=prompt_comment)
        return True

    def _export_prompt(self, *event):
        filename = Path(filedialog.asksaveasfilename(initialdir=self.default_path,
                                                   title="Select file",
                                                   filetypes=self.FILE_TYPE_PROMPT,))
        if filename.name:
            codecs.open(str(filename.absolute()), 'w', 'utf-8').write(self.prompt_retrieved.get('1.0', 'end-1c'))
            messagebox.showinfo('',
                                'Prompt Exported')

    def _export_prompts(self, *event):
        directory = Path(filedialog.askdirectory(initialdir=self.default_path,
                                                title="Select directory",
                                               ))
        if directory.name:
            for prompt_name in self.data.GetPromptNames():
                prompt = self.data.GetPrompt(prompt_name=prompt_name)
                try:
                    prompt_comment = self.data.GetCommentPrompt(prompt_name=prompt_name)
                except KeyError:
                    # print('no comment found for prompt {}'.format(prompt_name))
                    prompt_comment = 'not commented'

                filename = directory.joinpath(prompt_name+'.prompt')
                filename = str(filename.absolute())


                with codecs.open(filename, 'w', 'utf-8') as fout:
                    line_sep = self.EXPORT_PROMPT_COMMENT_BEGIN
                    fout.write(line_sep)
                    fout.write('\n')
                    # for line in prompt_comment:
                    #     line = '# {}\n'.format(line)
                    fout.writelines(prompt_comment)
                    #  write separator
                    line_sep = self.EXPORT_PROMPT_PROMPT_BEGIN
                    fout.write('\n')
                    fout.write(line_sep)
                    fout.write('\n')

                    for line in prompt:
                        fout.write(line)

            messagebox.showinfo('', 'Prompts Exported')

    def ShowHandleApiKeys(self, *event):
        def add_key(*event):
            self.api_keys[key_name.get()] = key_key.get()
            SaveJsonData(self.api_keys, self.api_keys_filename)
            keys_list.insert(0, key_name.get())

        def load_keys():
            keys_list.delete(0, 'end')
            for k in self.api_keys.keys():
                keys_list.insert(0, k)
            SaveJsonData(self.api_keys, self.api_keys_filename)

        def delete_key(key_name):
            self.api_keys.pop(key_name)
            SaveJsonData(self.api_keys, self.api_keys_filename)
            load_keys()

        def delete_key_selected(*event):
            if keys_list.curselection():
                k = keys_list.get(keys_list.curselection())
                delete_key(k)

        key_name = tk.StringVar()
        key_name.set('set-key-name')
        key_key = tk.StringVar()
        key_key.set('0000-0000--000-0000--000-000')

        kapp = tk.Toplevel(self.parent)
        #
        frm = tk.Frame(kapp)
        frm.pack(fill='both')
        #
        frm_name = tk.Frame(frm)
        frm_name.pack(side='top', fill='x')
        tk.Label(frm_name,
                 text='key name:').pack(side='left')
        tk.Entry(frm_name,
                 textvariable=key_name).pack(side='left')
        frm_key = tk.Frame(frm)
        frm_key.pack(side='top')
        tk.Label(frm_key,
                 text='key:').pack(side='left')
        tk.Entry(frm_key,
                 textvariable=key_key).pack(side='left')
        tk.Button(frm,
                  text='Add Key',
                  anchor='center',
                  command=add_key).pack(side='top', fill='x')
        frm_list=tk.Frame(frm)
        frm_list.pack(side='top')
        keys_list = tk.Listbox(frm_list)
        keys_list.pack(side='left')
        if self.api_keys:
            load_keys()
        else:
            self.api_keys = dict()
        #     messagebox.showwarning('',
        #                            'No Api Key found, you must set one')
        tk.Button(frm,
                  text='delete selected key',
                  command=delete_key_selected).pack(side='top', fill='x')

        kapp.mainloop()

    def ShowASingleAnswer_session(self, *event):
        #  get data
        def close():
            session_ans.destroy()

        if self.session.curselection():
            row = self.session.get(self.session.curselection())
            if row.startswith('Q: '):
                self.question_to_ask.delete('1.0', 'end')
                self.question_to_ask.insert('1.0', row[3:])
            else:
                answer_index = self.session.curselection()
                for index in range(answer_index[0]-1, -1, -1):
                    quest_row = self.session.get(index)
                    if quest_row.strip().startswith('Q:'):
                        question = quest_row.strip()[:55]
                        break
                session_ans = tk.Toplevel(self.parent)
                self.sessions_answer_objects.append(session_ans)
                session_ans.title(question)
                frm = tk.Frame(session_ans)
                frm.pack(side='top', fill='both', expand=True)
                t = tk.Text(frm,
                            height=23,
                            width=180,
                            wrap='word',)
                t.pack(side='left',
                       fill='both',
                       expand=True)
                ys = ttk.Scrollbar(frm,
                                   orient='vertical',
                                   command=t.yview)
                t['yscrollcommand'] = ys.set
                ys.pack(side='right', fill='y')

                xs = ttk.Scrollbar(session_ans,
                                   orient='horizontal',
                                   command=t.xview)
                t['xscrollcommand'] = xs.set
                xs.pack(side='top', fill='x')
                tk.Button(session_ans, text='Close', anchor='center', command=close).pack(side='top', fill='x')
                # insert data
                t.insert('1.0', row[3:])
                session_ans.mainloop()

    def get_parametrized_prompt_parameters_key(self,
                                               prompt: str):
        """
            a param key name is invalid if it contains spaces either at the beginning or at the end of the key name
        :param prompt:
        :return:
        """
        parameter_name = ''
        #  store parameters key name
        parameter_names = list()
        #  store invalid key name
        rejected_names = list()

        index = 0
        begin_index = 0
        look_for_begin = True
        # prompt = "is {x} greater than {y} or it is {something else}"
        while True:
            if look_for_begin:
                found_ind = prompt.find('{', index)
                # print(found_ind)
                if found_ind > 0:
                    begin_index = found_ind + 1
                    # move index
                    index = found_ind + 1
                    look_for_begin = False
                else:
                    break
            else:
                found_ind = prompt.find('}', index)
                # print(found_ind)
                if found_ind > 0:
                    # record name
                    parameter_name = prompt[begin_index:found_ind]
                    #  check the key name has not space at begin or end
                    if parameter_name == parameter_name.strip():
                        parameter_names.append(parameter_name)
                    else:
                        rejected_names.append(parameter_name)
                    # move index
                    index = found_ind + 1
                    look_for_begin = True
                else:
                    break

        return list(set(parameter_names)), list(set(rejected_names))

    def get_parametrized_questions_list_parameters_items(self,
                                                         parametrized_questions_list):
        def get_file_content(file_path):
            # return codecs.open(file_path, 'r', 'utf-8').read()
            try:
                return codecs.open(file_path, 'r', 'utf-8').read()
            except:
                print('file not found or file corrupted')
                return ''
        #  list of all the configurations
        all_params = list()
        # a single configuration
        params = dict()
        delim_pars = self.PARAMETRIZED_QUESTION_PARAMETERS_DELIM #'+++'
        assign_symbol = self.PARAMETRIZED_QUESTION_PARAMETERS_ASSIGN_SYMBOL #'@'
        record_param = False

        for line in parametrized_questions_list.split('\n'):
            # print(line)
            if line.strip().startswith(delim_pars):
                # print(record_param)
                if record_param:
                    # end of the block
                    all_params.append(params.copy())
                    # params.clear()
                    record_param = False
                else:
                    # begin of the block
                    record_param = True
            elif record_param:
                #  record key value setting
                try:
                    if line.strip() == '':
                        continue
                    key, value = line.split(assign_symbol)
                    # check if get file content or not
                    key = key.strip()
                    value = value.strip()
                    if value.startswith(self.PARAMETRIZED_QUESTION_PARAMETERS_FILE_CONTENT):
                        value = value.replace(self.PARAMETRIZED_QUESTION_PARAMETERS_FILE_CONTENT, '')
                        value = value.strip()
                        value = get_file_content(value)
                    params[key] = value
                except ValueError:
                    # # there are more than one assign_symbol ('@')
                    # the @ are assigned to key-value
                    # manual fix
                    # line = 'a@@f'
                    first_index = line.find(assign_symbol)
                    key = line[:first_index]
                    value = line[first_index+1:]
                    params[key.strip()] = value.strip()

        return all_params


def LunchPromptDesigner():
    root = tk.Tk()
    # frame = \
    PromptDesigner(root)
    root.mainloop()
    sys.exit(0)

if __name__ == '__main__':
    LunchPromptDesigner()
