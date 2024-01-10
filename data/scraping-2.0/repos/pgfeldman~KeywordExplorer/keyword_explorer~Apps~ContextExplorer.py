import re
import getpass
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as message
from datetime import datetime
from tkinter import filedialog
from enum import Enum
from pypdf import PdfReader

import pandas as pd

from keyword_explorer.Apps.AppBase import AppBase
from keyword_explorer.tkUtils.Buttons import Buttons
from keyword_explorer.tkUtils.ToolTip import ToolTip
from keyword_explorer.tkUtils.Checkboxes import Checkboxes
from keyword_explorer.tkUtilsExt.GPTContextFrame import GPTContextFrame, GPTContextSettings, PROMPT_TYPE
from keyword_explorer.tkUtils.ListField import ListField
from keyword_explorer.tkUtils.TextField import TextField
from keyword_explorer.tkUtils.DataField import DataField
from keyword_explorer.tkUtils.TopicComboExt import TopicComboExt
from keyword_explorer.tkUtils.LabeledParam import LabeledParam
from keyword_explorer.tkUtils.ListField import ListField
from keyword_explorer.OpenAI.OpenAIComms import OpenAIComms
from keyword_explorer.OpenAI.OpenAIEmbeddings import OpenAIEmbeddings
from keyword_explorer.utils.MySqlInterface import MySqlInterface
from keyword_explorer.utils.ManifoldReduction import ManifoldReduction, EmbeddedText, ClusterInfo
from keyword_explorer.utils.SharedObjects import SharedObjects

from typing import List, Dict

class CONTEXT_TEMPLATE(Enum):
    def __str__(self):
        return str(self.value)

    STORY = "Once upon a time there was"
    LIST = "Produce a list of items/concepts/phrases that are similar to '{}'|| first concept seed || second concept seed"
    SEQUENCE = "Produce the sequence of events that starts with '{}' and ends with '{}' || aaa && bbb || ccc && ddd"

class ContextExplorer(AppBase):
    oai: OpenAIComms
    oae: OpenAIEmbeddings
    msi: MySqlInterface
    mr: ManifoldReduction
    so:SharedObjects
    generator_frame: GPTContextFrame
    experiment_combo:TopicComboExt
    group_combo:TopicComboExt
    level_combo:TopicComboExt
    target_level_combo:TopicComboExt
    target_text_name:DataField
    target_group_field:DataField
    rows_field:DataField
    tokens_field:DataField
    keyword_filtered_field:DataField
    narrative_project_name_field:DataField
    generate_model_combo:TopicComboExt
    style_list:ListField
    action_buttons:Buttons
    action_buttons2:Buttons
    param_buttons:Buttons
    corpus_validation_text:TextField
    experiment_id_list:List


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("ContextExplorer")
        self.text_width = 60
        self.label_width = 15

        dt = datetime.now()
        experiment_str = "{}_{}_{}".format(self.app_name, getpass.getuser(), dt.strftime("%H:%M:%S"))
        self.experiment_field.set_text(experiment_str)
        self.load_group_list()
        self.load_experiment_list()
        self.experiment_id_list = []
        self.so.add_object("generate_model_combo", self.generate_model_combo, TopicComboExt)
        self.so.add_object("MySqlInterface", self.msi, MySqlInterface)
        # self.test_data_callback()

    def setup_app(self):
        self.app_name = "ContextExplorer"
        self.app_version = "11.21.2023"
        self.geom = (910, 820)
        self.oai = OpenAIComms()
        self.oae = OpenAIEmbeddings()
        self.so = SharedObjects()
        self.msi = MySqlInterface(user_name="root", db_name="gpt_summary")

        if not self.oai.key_exists():
            message.showwarning("Key Error", "Could not find Environment key 'OPENAI_KEY'")

        self.experiment_id = -1

    def build_app_view(self, row: int, text_width: int, label_width: int) -> int:
        print("build_app_view")
        self.generator_frame = GPTContextFrame(self.oai, self.dp, self.so)
        lf = tk.LabelFrame(self, text="GPT")
        lf.grid(row=row, column=0, columnspan = 2, sticky="nsew", padx=5, pady=2)
        self.build_gpt(lf, text_width, label_width)

        lf = tk.LabelFrame(self, text="Params")
        lf.grid(row=row, column=2, sticky="nsew", padx=5, pady=2)
        self.build_params(lf, int(text_width/3), int(label_width*.75))
        return row + 1

    def load_group_list(self):
        experiments = ['All Groups']
        groups = self.msi.read_data("select distinct group_name from table_source order by group_name")
        for g in groups:
            group_name = g['group_name']
            experiments.append(group_name)
        self.group_combo.set_combo_list(experiments)
        self.group_combo.set_text("All Groups")

    def load_experiment_list(self, group_name = "All Groups"):

        if group_name != "All Groups":
            sql = "select * from table_source where group_name = %s order by text_name"
            results = self.msi.read_data(sql, (group_name,))
        else: # All Groups
            results = self.msi.read_data("select * from table_source order by group_name, text_name")

        entries = self.msi.read_data("select group_name, count(group_name) as entries from table_source group by group_name")
        prev_name = "NO-PREV-NAME"
        experiments = []
        for r in results:
            group_name = r['group_name']
            for e in entries:
                if e['group_name'] == group_name and e['entries'] > 1 and prev_name != group_name:
                    experiments.append("*:{}".format(group_name))
            experiments.append("{}:{}".format(r['text_name'], group_name))
            prev_name = group_name
        self.experiment_combo.set_combo_list(experiments)

    def get_levels_list(self):
        level_list = ['all', 'raw only', 'all summaries']
        if self.experiment_id != -1:
            sql = "select distinct level from table_summary_text where source = %s"
            vals = self.experiment_id
            results = self.msi.read_data(sql, vals)
            d:Dict
            for d in results:
                level_list.append(d['level'])
        self.level_combo.set_combo_list(level_list)
        self.level_combo.tk_combo.current(0)
        self.level_combo.clear()
        self.level_combo.set_text(self.level_combo.get_list()[0])

    def build_gpt(self, lf:tk.LabelFrame, text_width:int, label_width:int):
        row = 0
        self.group_combo = TopicComboExt(lf, row, "Project Groups:", self.dp, entry_width=20, combo_width=30)
        self.group_combo.set_callback(self.set_group_callback)
        row = self.group_combo.get_next_row()
        self.experiment_combo = TopicComboExt(lf, row, "Saved Projects:", self.dp, entry_width=20, combo_width=30)
        self.experiment_combo.set_callback(self.load_project_callback)
        row = self.experiment_combo.get_next_row()
        self.level_combo = TopicComboExt(lf, row, "Summary Levels:", self.dp, entry_width=20, combo_width=30)
        self.level_combo.set_callback(self.count_levels_callback)
        row = self.level_combo.get_next_row()
        self.narrative_project_name_field = DataField(lf, row, "NarrativeMap",text_width, label_width=label_width)
        ToolTip(self.narrative_project_name_field.tk_entry, "Project name to export NarrativeMaps")
        row = self.narrative_project_name_field.get_next_row()

        self.action_buttons = Buttons(lf, row, "Actions")
        b = self.action_buttons.add_button("Load Data", self.load_data_callback, width=-1)
        ToolTip(b, "Load data for selected project")
        b = self.action_buttons.add_button("Export", self.save_to_narrative_maps_jason_callback, width=-1)
        ToolTip(b, "Export Project to JSON")
        row = self.action_buttons.get_next_row()

        engine_list = self.oai.list_models(exclude_list = [":", "tts", "vision", "ada", "embed", "similarity", "code", "edit", "search", "audio", "instruct", "2020", "if", "insert", "whisper"])
        engine_list = sorted(engine_list, reverse=True)
        self.generate_model_combo = TopicComboExt(lf, row, "Model:", self.dp, entry_width=25, combo_width=25)
        self.generate_model_combo.set_combo_list(engine_list)
        self.generate_model_combo.set_text(engine_list[0])
        self.generate_model_combo.tk_combo.current(0)
        ToolTip(self.generate_model_combo.tk_combo, "The GPT-3 model used to generate text")
        row = self.generate_model_combo.get_next_row()

        s = ttk.Style()
        s.configure('TNotebook.Tab', font=self.default_font)

        # Add the tabs
        tab_control = ttk.Notebook(lf)
        tab_control.grid(column=0, row=row, columnspan=2, sticky="nsew")
        gpt_tab = ttk.Frame(tab_control)
        tab_control.add(gpt_tab, text='Output')
        self.build_generator_tab(gpt_tab, text_width, label_width)

        corpora_tab = ttk.Frame(tab_control)
        tab_control.add(corpora_tab, text='Corpora')
        self.build_corpora_tab(corpora_tab, text_width, label_width)

        row += 1
        return row

    def build_params(self, lf:tk.LabelFrame, text_width:int, label_width:int):
        row = 0
        self.rows_field = DataField(lf, row, 'Rows:', text_width, label_width=label_width)
        row = self.rows_field.get_next_row()
        self.keyword_filtered_field = DataField(lf, row, 'Filtered:', text_width, label_width=label_width)
        row = self.keyword_filtered_field.get_next_row()
        self.tokens_field = DataField(lf, row, 'Tokens:', text_width, label_width=label_width)
        self.tokens_field.set_text("1024")
        row = self.tokens_field.get_next_row()

        self.style_list = ListField(lf, row, "Style\n({})".format(PROMPT_TYPE.NARRATIVE.value), width=text_width, label_width=label_width, static_list=True)
        # self.style_list.set_text(text='Story, List, Sequence')
        self.style_list.clear()
        self.style_list.add_entry(PROMPT_TYPE.NARRATIVE.value)
        self.style_list.add_entry(PROMPT_TYPE.LIST.value)
        self.style_list.add_entry(PROMPT_TYPE.SEQUENCE.value)
        # self.style_list.set_callback(self.set_style_callback)
        ToolTip(self.style_list.tk_list, "Select the mode of behavior that the prompt will generate")
        row = self.style_list.get_next_row()
        self.style_list.tk_list.select_set(0)

        self.param_buttons = Buttons(lf, row, 'Actions', label_width=label_width)
        b = self.param_buttons.add_button("Set Style", self.set_style_callback)
        ToolTip(b, "Selects the mode from the list above.\nA hack to avoid a list callback bug")
        row = self.param_buttons.get_next_row()


    def build_generator_tab(self, tab: ttk.Frame, text_width:int, label_width:int):
        self.generator_frame.build_frame(tab, text_width, label_width)

    def build_corpora_tab(self, tab: ttk.Frame, text_width:int, label_width:int):
        row = 0

        self.target_text_name = DataField(tab, row, "Target Name")
        row = self.target_text_name.get_next_row()
        self.target_group_field = DataField(tab, row, "Target Group")
        row = self.target_group_field.get_next_row()

        target_level_list = [1, 2, 3, 4]
        self.target_level_combo = TopicComboExt(tab, row, "Target Summary Level:", self.dp, entry_width=20, combo_width=20)
        self.target_level_combo.set_combo_list(target_level_list)
        self.target_level_combo.tk_combo.current(0)
        self.target_level_combo.clear()
        self.target_level_combo.set_text(target_level_list[0])
        row = self.target_level_combo.get_next_row()

        self.regex_field = DataField(tab, row, 'Parse regex:', text_width, label_width=label_width)
        self.regex_field.set_text(r"([\.!?()]+)")
        ToolTip(self.regex_field.tk_entry, "The regex used to parse the file. Editable")
        row = self.regex_field.get_next_row()

        self.action_buttons2 = Buttons(tab, row, "Actions")
        b = self.action_buttons2.add_button("Test", self.test_file_callback, width=-1)
        ToolTip(b, "Performs a small test on 10 lines of text and does not save to DB")
        b = self.action_buttons2.add_button("Load File", self.load_file_callback, width=-1)
        ToolTip(b, "Loads new text into a project, splits into chunks and finds embeddings")
        b = self.action_buttons2.add_button("Validate File", self.validate_file_callback, width=-1)
        ToolTip(b, "Opens and reads the file and shows the result below")

        row = self.action_buttons2.get_next_row()
        self.corpus_validation_text = TextField(tab, row, "Valid Text?", text_width, height=11, label_width=label_width)

    def set_style_callback(self, event:tk.Event = None):
        print("ContextExplorer.set_style_callback(): event = {}".format(event))
        buttons:Buttons = self.so.get_object("context_buttons")
        style_str = self.style_list.get_selected()
        if style_str == "unset":
            # We have a bad call, so don't do anything more
            return
        buttons.change_button_label(PROMPT_TYPE.NARRATIVE.value, style_str)
        self.style_list.set_label("Style\n({})".format(style_str))
        if style_str == PROMPT_TYPE.NARRATIVE.value:
            self.generator_frame.prompt_text_field.set_text(CONTEXT_TEMPLATE.STORY.value)
        elif style_str == PROMPT_TYPE.LIST.value:
            self.generator_frame.prompt_text_field.set_text(CONTEXT_TEMPLATE.LIST.value)
        elif style_str == PROMPT_TYPE.SEQUENCE.value:
            self.generator_frame.prompt_text_field.set_text(CONTEXT_TEMPLATE.SEQUENCE.value)
            print("Set Sequence regex")
        self.set_narrative_name()

    def set_group_callback(self, event = None):
        print("set_group_callback")
        s = self.group_combo.tk_combo.get()
        self.group_combo.clear()
        self.group_combo.set_text(s)
        self.load_experiment_list(s)

    def load_project_callback(self, event = None):
        print("load_project_callback")
        s = self.experiment_combo.tk_combo.get()
        l = s.split(":")
        self.experiment_combo.clear()
        self.experiment_combo.set_text(s)

        #If there is a "*" for text_name, then just search for group
        text_name = l[0]
        sql = "select id from table_source where text_name = %s and group_name = %s"
        vals = (text_name,l[1])
        if text_name == "*":
            sql = "select id from table_source where group_name = %s"
            vals = (l[1],)
        results = self.msi.read_data(sql, vals)
        if len(results) > 0:
            self.experiment_id_list = []
            self.experiment_id = results[0]['id']
            self.experiment_field.set_text(" experiment {}: {}".format(self.experiment_id, s))
            self.set_narrative_name()
            self.target_group_field.set_text(l[1])
            # load up all the experiment id's
            for r in results:
                self.experiment_id_list.append(r['id'])

        self.get_levels_list()
        self.count_levels_callback()
        print("\tload_project_callback: experiment_id = {}/{}".format(self.experiment_id, self.experiment_id_list))

    def count_levels_callback(self, event = None):
        print("ContextExplorer.count_levels_callback(): ")
        if self.experiment_id == -1:
            tk.messagebox.showwarning("Warning!", "Please create or select a database first")
            return

        level = self.level_combo.tk_combo.get()
        self.level_combo.clear()
        self.level_combo.set_text(level)
        print("\tlevel = '{}'".format(level))

        raw_count = 0
        summary_count = 0
        if level == 'raw only' or level == 'all':
            print("\t'raw only' or 'all'")
            sql = "select count(*) from gpt_summary.table_parsed_text where source = {}".format(self.experiment_id)
            if len(self.experiment_id_list) > 0:
                sql =  "select count(*) from gpt_summary.table_parsed_text where source in ({})".format(", ".join(map(str, self.experiment_id_list)))
            print("\t{}".format(sql))
            results = self.msi.read_data(sql)
            raw_count = int(results[0]['count(*)'])
            self.rows_field.set_text("{:,}".format(raw_count))
        if level == 'all summaries' or level == 'all':
            print("\t'all summaries' or 'all'")
            sql = "select count(*) from gpt_summary.table_summary_text where source = {}".format(self.experiment_id)
            if len(self.experiment_id_list) > 0:
                sql =  "select count(*) from gpt_summary.table_summary_text where source in ({})".format(", ".join(map(str, self.experiment_id_list)))
            print("\t{}".format(sql))
            results = self.msi.read_data(sql)
            summary_count = int(results[0]['count(*)'])
            self.rows_field.set_text("{:,}".format(summary_count))
        if level == 'all':
            print("\t'all'")
            self.rows_field.set_text("{:,}".format(raw_count + summary_count))

        try:
            level = int(level)
            print("\tlevel {}".format(level))
            sql = "select count(*) from gpt_summary.table_summary_text where level = {} and source = {}".format(level, self.experiment_id)
            if len(self.experiment_id_list) > 0:
                sql =  "select count(*) from gpt_summary.table_summary_text where level = {} and source in ({})".format(level, ", ".join(map(str, self.experiment_id_list)))
            print("\t{}".format(sql))
            results = self.msi.read_data(sql)
            count = int(results[0]['count(*)'])
            self.rows_field.set_text("{:,}".format(count))
        except ValueError:
            pass

    def load_data_callback(self, event = None):
        max_tokens = self.tokens_field.get_as_int()
        self.generator_frame.set_max_tokens(max_tokens)

        kw_list = []
        kw_str = self.generator_frame.keyword_filter.get_text()
        if len(kw_str) > 3:
            s:str
            kw_list = kw_str.split("OR")
            kw_list = [s.strip() for s in kw_list]
        print("load_data_callback")
        df = pd.DataFrame()
        if self.experiment_id == -1:
            tk.messagebox.showwarning("Warning!", "Please create or select a database first")
        level = self.level_combo.tk_combo.get()
        df_list = []
        if level == 'raw only' or level == 'all':
            sql = "select text_id, parsed_text, embedding from source_text_view where source_id = {}".format(self.experiment_id)
            if len(self.experiment_id_list) > 0:
                print("ContextExplorer.load_data_callback(): loading raw data for sources {}".format(self.experiment_id_list))
                sql = "select text_id, parsed_text, embedding from source_text_view where source_id in ({})".format(", ".join(map(str, self.experiment_id_list)))
            if len(kw_list) > 0:
                sql += " AND (parsed_text LIKE '%{}%')".format("%' OR parsed_text LIKE '%".join(kw_list))
            results = self.msi.read_data(sql)
            df = self.oae.results_to_df(results)
            df_list.append(df)
        if level == 'all summaries' or level == 'all':
            sql = "select text_id, parsed_text, embedding, origins from summary_text_view where proj_id = {}".format(self.experiment_id)
            if len(self.experiment_id_list) > 0:
                print("ContextExplorer.load_data_callback(): loading all summaries for sources {}".format(self.experiment_id_list))
                sql = "select text_id, parsed_text, embedding, origins from summary_text_view where proj_id in ({})".format(", ".join(map(str, self.experiment_id_list)))
            if len(kw_list) > 0:
                sql += " AND (parsed_text LIKE '%{}%')".format("%' OR parsed_text LIKE '%".join(kw_list))
            results = self.msi.read_data(sql)
            df = self.oae.results_to_df(results)
            df_list.append(df)
        if level == 'all':
            df = pd.concat(df_list, ignore_index=True)

        try:
            level = int(level)
            print("level {}".format(level))
            sql = "select text_id, parsed_text, embedding, origins from summary_text_view where level = {} and proj_id = {}".format(level, self.experiment_id)
            if len(self.experiment_id_list) > 0:
                print("ContextExplorer.load_data_callback(): loading level {} for sources {}".format(level, self.experiment_id_list))
                sql = "select text_id, parsed_text, embedding, origins from summary_text_view where level = {} and proj_id in ({})".format(level, ", ".join(map(str, self.experiment_id_list)))

            if len(kw_list) > 0:
                sql += " AND (parsed_text LIKE '%{}%')".format("%' OR parsed_text LIKE '%".join(kw_list))
                print(sql)
            results = self.msi.read_data(sql)
            df = self.oae.results_to_df(results)
        except ValueError:
            pass

        self.generator_frame.set_project_dataframe(df)
        self.keyword_filtered_field.set_text("{:,}".format(len(df.index)))

        self.generator_frame.clear_callback(clear_keywords=False)
        max_tokens = self.tokens_field.get_as_int()
        self.generator_frame.auto_question_callback()

    def get_current_params(self) -> Dict:
        d = self.generator_frame.get_settings().to_dict()
        d['name'] = self.experiment_field.get_text()
        d['narrative-name'] = self.narrative_project_name_field.get_text()
        return d

    def load_experiment_callback(self, event = None):
        print("ContextExplorer:load_experiment_callback")
        defaults = self.get_current_params()

        param_dict = self.load_json(defaults)
        print("param_dict = {}".format(param_dict))

        gs = GPTContextSettings()
        gs.from_dict(param_dict)
        self.generator_frame.set_params(gs)

        self.experiment_field.clear()
        self.experiment_field.set_text(param_dict['name'])

        self.narrative_project_name_field.clear()
        self.narrative_project_name_field.set_text(param_dict['narrative-name'])

    def set_narrative_name(self):
        name_str = self.experiment_combo.tk_combo.get()
        buttons:Buttons = self.so.get_object("context_buttons")
        style_str = buttons.get_button_label(PROMPT_TYPE.NARRATIVE.value)
        s ="experiment {}: {}_{}".format(self.experiment_id, name_str, style_str)
        self.narrative_project_name_field.set_text(s)
        print("ContextExplorer.set_narrative_name(): [{}]".format(s))

    def save_to_narrative_maps_jason_callback(self, event = None):
        print("save_to_narrative_maps_callback")
        if self.experiment_id == -1:
            tk.messagebox.showwarning("Warning!", "Please create or select a database first")
            return
        buttons:Buttons = self.so.get_object("context_buttons")
        style_str = buttons.get_button_label(PROMPT_TYPE.NARRATIVE.value)

        probe_str =  "{}".format(self.generator_frame.prompt_text_field.get_text())
        context_str =  "{}".format(self.generator_frame.context_text_field.get_text())
        name = self.narrative_project_name_field.get_text()
        type = self.style_list.get_selected()
        regex_str = self.generator_frame.story_response_regex.pattern
        if style_str == PROMPT_TYPE.LIST.value:
            regex_str = self.generator_frame.list_response_regex.pattern
        elif style_str == PROMPT_TYPE.SEQUENCE.value:
            regex_str = self.generator_frame.sequence_response_regex.pattern
        dict = {"probe_str": probe_str, "context":context_str, "name":name, "type":type, "regex_str":regex_str}
        self.save_experiment_json(dict)

    def read_input_file(self, regex_str:str) -> [str, List[str]]:
        result = filedialog.askopenfilename(filetypes=(("Text and pdf files", "*.txt *.pdf"),("All Files", "*.*")), title="Load text file")
        textfile = result.split("/")[-1]
        s:str
        if result:
            s_list = []
            if result.endswith(".pdf"):
                s_list = self.oae.parse_pdf_file(result, r_str=regex_str)
            elif result.endswith(".txt"):
                s_list = self.oae.parse_text_file(result, r_str=regex_str)
            else:
                print("ContextExplorer.load_file_callback: unable to open file {}".format(result))
                return ("NONE", [])

            # clean out things we don't want to pay for
            regexp = re.compile(r'bibliography:?\n|references:?\n')
            for i in range(len(s_list)):
                s = s_list[i]
                if regexp.search(s):
                    print("ContextExplorer.load_file_callback(): truncating references (rows {} - {})".format(i-1, len(s_list)))
                    s_list = s_list[:i-1]
                    break

            self.corpus_validation_text.set_text("\n".join(s_list))
            return(textfile, s_list)
        return ("NONE", [])

    def validate_file_callback(self, event = None):
        regex_str = self.regex_field.get_text()
        textfile, s_list = self.read_input_file(regex_str)

    def load_file_callback(self, event = None):
        print("ContextExplorer.load_file_callback()")
        group_name = self.target_group_field.get_text().strip()
        text_name = self.target_text_name.get_text().strip()
        regex_str = self.regex_field.get_text()
        if len(group_name) < 3 or len(text_name) < 3:
            tk.messagebox.showwarning("Warning!", "Please set text and model fields")
            return
        textfile, s_list = self.read_input_file(regex_str)
        s:str
        if textfile != "NONE":

            answer = tk.messagebox.askyesno("Warning!", "This will read, process, and store large amounts of data\ntarget = [{}]\ngroup = [{}]\nfile = [{}]\nlines = [{:,}]\nProceed?".format(
                text_name, group_name, textfile, len(s_list)))
            if answer == True:
                engine = self.generate_model_combo.get_text()
                level = int(self.target_level_combo.get_text())
                print("ContextExplorer.load_file_callback(): Getting embeddings")
                df = self.oae.get_embeddings(s_list)
                print("ContextExplorer.load_file_callback(): Storing data Dataframe = \n{}".format(df))
                self.oae.store_project_data(text_name, group_name, df)
                print("ContextExplorer.load_file_callback(): Summarizing Level 1")
                max_tokens = self.tokens_field.get_as_int()
                self.oae.summarize_raw_text(text_name, group_name, engine=engine, max_tokens=max_tokens)
                for i in range(1, level):
                    print("ContextExplorer.load_file_callback(): Summarizing Level {}".format(i+1))
                    self.oae.summarize_summary_text(text_name, group_name, source_level=i)
                print("ContextExplorer.load_file_callback(): Getting summary embeddings")
                self.oae.set_summary_embeddings(text_name, group_name)
                print("ContextExplorer.load_file_callback(): Finished!")

    def test_file_callback(self, event = None):
        print("ContextExplorer.test_file_callback()")
        group_name = self.target_group_field.get_text()
        text_name = self.target_text_name.get_text()
        regex_str = self.regex_field.get_text()
        if len(group_name) < 3 or len(text_name) < 3:
            tk.messagebox.showwarning("Warning!", "Please set text and model fields")
            return
        textfile, s_list = self.read_input_file(regex_str)
        if textfile != "NONE":
            s:str
            num_rows = 10

            # build a proxy db results List
            results = []
            for i in range(num_rows):
                s = s_list[i]
                d = {"text_id":i, "parsed_text":s}
                results.append(d)
            row_dict:Dict
            count = 0
            words_to_summarize = 300
            #engine = self.oae.DEFAULT_SUMMARY_MODEL
            engine = self.generate_model_combo.get_text()
            print("\tUsing {}".format(engine))
            max_tokens = self.tokens_field.get_as_int()
            while count < num_rows:
                d = self.oae.build_text_to_summarize(results, count, words_to_summarize, overlap=2)
                # run the query and store the result. Update the parsed text table with the summary id
                summary = self.oai.get_prompt_result_params(d['query'], engine=engine, temperature=0, presence_penalty=0.8, frequency_penalty=0, max_tokens=max_tokens)
                if summary == self.oai.ERROR_MSG:
                    print("\ttest_file_callback() got {} from self.oai.get_prompt_result_params({})".format(summary, d['query']))
                    continue

                embd = self.oai.get_embedding_list([summary])
                mod = self.oai.get_moderation_vals([summary])
                print("\tSummary[{}]: {}\n\tEmbedding: {}\n\tSpeech: {}\n".format(count, summary, embd[0]['embedding'], mod[0]['category_scores']))

                if count == d['count']:
                    print("\t forcing count increment")
                    count += 1
                else:
                    count = d['count']

        print("\ttest_file_callback(): Complete")


def main():
    app = ContextExplorer()
    app.mainloop()

if __name__ == "__main__":
    main()