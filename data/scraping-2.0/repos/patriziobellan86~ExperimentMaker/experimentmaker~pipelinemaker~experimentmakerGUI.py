import codecs
import tkinter as tk
import tkinter.ttk as ttk
import os
import openai
from tkinter import messagebox
from tkinter import filedialog
from pathlib import Path
from promptdesignerdataset.dataset import PromptDesignerDataset, LoadJsonData
from PIL import Image, ImageTk
from pipelinemaker.pipeline import Pipeline


# for stand-alone
# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
#     return os.path.join(base_path, relative_path)

# for pip
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = os.path.dirname(os.path.abspath(__file__)) # getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class ExperimentGUI: #(tk.Frame):
    __version__ = '1.0.1'
    PADX = 5
    PADY = 2
    IMG_X = 25
    BUTTON_HEIGHT = IMG_X + 5

    DO_NOTHING = 'Do nothing'
    #
    # FOR_EACH_ELEMENT = 'for-each-element'
    # FOR_EACH_PAIR = 'for-each-pair'
    #
    # OUTPUT_LIST_OUTPUT = 'output-list'
    # OUTPUT_LIST_OUTPUT = 'output-list'
    # OUTPUT_SINGLE_OUTPUT = 'output-single'
    #  apply filter on the results
    ITEM_OUTPUT_FILTER_TYPE_RESULTS = 'filter on results'
    #  apply filter on the item
    ITEM_OUTPUT_FILTER_TYPE_ITEM = 'ad-hoc filter'

    def __init__(self,
                 parent,
                 ):
        # tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parent.geometry("1200x700")
        self.parent.title('Prompt Designer - Experiments Maker')

        self.__init_variables__()
        self.__load_icons()
        self.__load_program_icon()
        self.CreateFrames()
        self.CreateMenu()

        self._fill_prompts_combobox()
        self._fill_pipeline_item_output_filter_list()

    def __load_program_icon(self):
        # return
        self.parent.tk.call('wm', 'iconphoto', self.parent._w, self.images['logo'])


    def __init_variables__(self):
        #  handle prompts
        self.promptdesignerdataset = PromptDesignerDataset()
        # item output filter type
        self.item_output_filter_type = tk.StringVar()

        #  store module imported
        self.imported_output_filters = dict()
        self.imported_item_output_filtes = dict()

        self.pipeline_name = tk.StringVar()
        self.pipeline_name.trace_variable(mode='w', callback=self._pipeline_name_changed)

        #  store input file filename
        self.input_file_filename = None
        #  store input type
        self.input_type = tk.StringVar()
        #  store mnemonic name of the input file
        self.input_file_mnemonic_name = tk.StringVar()

        self.selected_setting_filename = tk.StringVar()
        self._prompt_key_name = tk.StringVar()
        self._prompt_input_value = tk.StringVar()
        #  opened input file content
        self.input_text_content = None

        #  dataset completed path
        self.output_pipeline_filename = tk.StringVar()

        #  selected prompt
        self.selected_prompt = tk.StringVar()

        #  pipiline item name
        self.pipeline_item_name = tk.StringVar()

        # handle radio button selection
        #  keep radio vaule for each element or for each pair
        self.experiment_setting_range_type = tk.StringVar()
        #  keep radio value for output type
        self.experiment_output_type = tk.StringVar()

        self.chk_apply_filter_to_pipeline_output = tk.BooleanVar()
        self.chk_apply_filter_to_pipeline_output.set(False)

        self.pipeline_item_output_filter_name = tk.StringVar()
        self.pipeline_output_filter_name = tk.StringVar()

        #  GPT-3 Settings
        self.gpt3settings = dict()

        #  Pipeline
        self.pipeline = Pipeline()

        #  checked status for re-run the entire pipeline or only the missing steps
        self._force_rerun_entire_pipeline = tk.BooleanVar()
        self._force_rerun_entire_pipeline.set(True)

    def __load_icons(self):
        # icon_folder = Path(__file__).absolute()
        # icon_folder = icon_folder.parent
        icon_folder = Path(resource_path('icons'))
        # print(os.path.exists(icon_folder))
        # print(os.path.exists(icon_folder.joinpath('load_file.png')))

        self.images = {'load': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('load_file.png')).resize((self.IMG_X, self.IMG_X)),
                       master=self.parent),
                       'save': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('save_file.png')).resize((self.IMG_X, self.IMG_X)),
                       master=self.parent),
                       'show': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('show.png')).resize((self.IMG_X, self.IMG_X)),
                       master=self.parent),
                       'clear': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('clear.png')).resize((self.IMG_X, self.IMG_X)),
                       master=self.parent),

                       'pipeline': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('pipeline.png')).resize((self.IMG_X, self.IMG_X)),
                       master=self.parent),
                       'experiment': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('experiment.png')).resize((self.IMG_X, self.IMG_X)),
                       master=self.parent),
                       'add': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('add.png')).resize((self.IMG_X, self.IMG_X)),
                       master=self.parent),
                       'checked': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('checked.png')).resize((self.IMG_X, self.IMG_X)),
                       master=self.parent),

                        'logo': ImageTk.PhotoImage(
                            Image.open(icon_folder.joinpath('logo.png')).resize((self.IMG_X, self.IMG_X)),
                                master=self.parent),
                       }
        #
        # self.images = {'load': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('load_file.png')).resize((self.IMG_X, self.IMG_X))),
        #                'save': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('save_file.png')).resize((self.IMG_X, self.IMG_X))),
        #                'show': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('show.png')).resize((self.IMG_X, self.IMG_X))),
        #                'clear': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('clear.png')).resize((self.IMG_X, self.IMG_X))),
        #
        #                'pipeline': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('pipeline.png')).resize((self.IMG_X, self.IMG_X))),
        #                'experiment': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('experiment.png')).resize((self.IMG_X, self.IMG_X))),
        #                'add': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('add.png')).resize((self.IMG_X, self.IMG_X))),
        #                'checked': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('checked.png')).resize((self.IMG_X, self.IMG_X))),
        #
        #                 'logo': ImageTk.PhotoImage(
        #                     Image.open(icon_folder.joinpath('logo.png')).resize((self.IMG_X, self.IMG_X))),
        #                }

    def CreateMenu(self):
        self.menubar = tk.Menu(self.parent)

        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Import prompts from prompt dataset", command=self.LoadPromptDesignerDataset)

        self.filemenu.add_separator()
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        gptsettings = tk.Menu(self.menubar, tearoff=0)
        gptsettings.add_command(label="Load GPT-3 settings", command=self._load_gptsettings)
        gptsettings.add_command(label="Show GPT-3 settings", command=self._show_gpt3_settings)
        self.filemenu.add_cascade(label="GPT-3 Settings", menu=gptsettings)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.parent.destroy)

        pipeline = tk.Menu(self.menubar, tearoff=0)
        pipeline.add_command(label="New", command=self.NewPipeline)
        pipeline.add_command(label="Load", command=self.LoadPipeline)
        pipeline.add_command(label="Save", command=self.SavePipeline)
        pipeline.add_command(label="Save As", command=self.SaveAsPipeline)
        pipeline.add_separator()
        pipeline.add_command(label="Clear", command=self.ClearPipeline)
        pipeline.add_separator()
        pipeline.add_command(label='Show item', command=self.ShowPipelineItem)
        self.menubar.add_cascade(label="Pipeline", menu=pipeline)

        results = tk.Menu(self.menubar, tearoff=0)
        results.add_command(label="Show results", command=self.ShowResults)
        results.add_command(label="Export results", command=self.ExportResultsGUI)
        self.menubar.add_cascade(label="Results", menu=results)

        filters = tk.Menu(self.menubar, tearoff=0)
        filters.add_command(label="Import item-filter module", command=self.LoadItemOutputModuleGUI)
        filters.add_command(label="Import output script module", command=self.LoadOutputFilterModuleGUI)
        self.menubar.add_cascade(label="Filters and Scripts", menu=filters)

        self.lunch = tk.Menu(self.menubar, tearoff=0)
        self.lunch.add_command(label="Lunch Experiment", command=self.LunchExperimentalPipeline)
        self.lunch.add_command(label="Apply output script", command=self.LunchOutputScript)
        self.lunch.add_separator()
        self.lunch.add_checkbutton(label="force re-run the entire pipeline", onvalue=1, offvalue=0, variable=self._force_rerun_entire_pipeline)
        self.lunch.add_separator()
        self.lunch.add_command(label='run only selected item', command=self.LunchExperimentalPipelineSelectedItem)

        self.menubar.add_cascade(label="Lunch Experiment", menu=self.lunch)

        self.parent.config(menu=self.menubar)
    #
    # def _force_rerun_entire_pipeline_change_status(self):
    #     if self._force_rerun_entire_pipeline.get():
    #         #  set status
    #         self._force_rerun_entire_pipeline.set(False)
    #         #  set icon on menu
    #         self.lunch.entryconfig(3, {'image': None})
    #     else:
    #         #  set status
    #         self._force_rerun_entire_pipeline.set(True)
    #         #  set icon on menu
    #         self.lunch.entryconfig(3, {'image': self.images['checked']})

    def _add_input_type_item(self, input_item):
        if type(self.cmb_input_prompt_input_value['values']) == str:
            self.cmb_input_prompt_input_value['values'] = [input_item]
        else:
            vals = list(self.cmb_input_prompt_input_value['values'])
            vals.append(input_item)
            vals = list(set(vals))
            self.cmb_input_prompt_input_value['values'] = vals

    def CreateFrames(self):
        #   MAIN TOP FRAME
        self.frm_maintop = tk.Frame(self.parent)
        self.frm_maintop.pack(fill='x',
                              # expand=True,
                              side='top')
        self.__create_frame_maintop()

        #   MAIN PIPELINE FRAME
        self.frm_main_pipeline = tk.Frame(self.parent)
        self.frm_main_pipeline.pack(fill='both',
                               expand=True,
                               side='top')
        self.__create_frame_mainpipeline()

    def _clear_output_script(self, *event):
        self.lst_output_script.delete(0, 'end')

    def _add_item_output_filter(self, *event):
        if self.pipeline_item_output_filter_name.get():
            filter_item = tuple([self.pipeline_item_output_filter_name.get(), self.item_output_filter_type.get()])
            self.lst_item_filters.insert('end', filter_item)

    def _fill_pipeline_item_output_filter_list(self):
        self.cmb_pipeline_item_output_filter_name['values'] = list()

    def __create_frame_maintop(self):
        self.__frm_maintop_left = tk.Frame(self.frm_maintop,
                                           width=200,
                                           )
        self.__frm_maintop_left.pack(side='left', fill='both')
        frm_pipeline_name = tk.Frame(self.__frm_maintop_left)
        frm_pipeline_name.pack(side='top', fill='x')
        tk.Label(frm_pipeline_name,
                 text='Pipeline name:',
                 anchor='w',
                 ).pack(side='left')
        tk.Entry(frm_pipeline_name,
                 textvariable=self.pipeline_name,
                 justify='center',
                 ).pack(side='left', fill='x')
        self.__create_frame_maintop_inputfile()
        # self.__create_frame_maintop_gptsettings()
        self.__create_frame_maintop_output()

        self.__frm_maintop_experimentitem = tk.LabelFrame(self.frm_maintop,
                                                          text='Experimental Item',
                                                          labelanchor='n')
        self.__frm_maintop_experimentitem.pack(side='left',
                                               fill='both',
                                               expand=True)
        self.__create_frame_experimentitem()

    def _clear_input_key_pairs(self):
        for item in self.input_key_pairs.get_children():
            self.input_key_pairs.delete(item)

    def _clear_input_values(self):
        self.cmb_input_prompt_input_value['values'] = []
        self.cmb_input_prompt_input_value.set('')

        for menomnic_input_name in self.pipeline.GetInputFilesMnemonicNames():
            self._add_input_type_item(menomnic_input_name)

    def _add_inputfile_into_filetreelist(self,
                                         filename_name,
                                         mnemonic_name,
                                         inputtype):

        self.input_files_tree.insert('',
                                     'end',
                                     text=mnemonic_name,
                                     values=([mnemonic_name, inputtype]))
        self._add_input_type_item(mnemonic_name)

    def __create_frame_maintop_inputfile(self):
        frm_input = tk.LabelFrame(self.__frm_maintop_left,
                                      text='Input Files',
                                      labelanchor='n')
        frm_input.pack(side='top', fill='x', expand=True)
        frm_inputbuttons = tk.Frame(frm_input)
        frm_inputbuttons.pack(side='top',
                              fill='x')
        tk.Button(frm_inputbuttons,
                  text='', #  Select Input File',
                  anchor='w',
                  height=self.BUTTON_HEIGHT,
                  width=self.BUTTON_HEIGHT,
                  image=self.images['load'],
                  compound='center',
                  relief='raised',
                  command=self.SelectInputFile,
                  ).pack(side='left')

        tk.Button(frm_inputbuttons,
                  text='', #Show text',
                  anchor='w',
                  image=self.images['show'],
                  compound='center',
                  height=self.BUTTON_HEIGHT,
                  width=self.BUTTON_HEIGHT,
                  command=self.InputFilePreview,
                  ).pack(side='left')

        tk.Label(frm_inputbuttons, #frm_input_selectedfile,
                 anchor='w',
                 # relief='raised',
                 text='Mnemonic Name',
                 ).pack(side='left')
        tk.Entry(frm_inputbuttons, #frm_input_selectedfile,
                 justify='left',
                 width=15,
                 relief='raised',
                 textvariable=self.input_file_mnemonic_name,
                 ).pack(side='left', fill='x')

        frm_input_type = tk.Frame(frm_input)
        frm_input_type.pack(side='top', fill='x')
        tk.Radiobutton(frm_input_type,
                       text='Entire file',
                       anchor='w',
                       value=Pipeline.INPUT_TYPE_ENTIRE,
                       variable=self.input_type,
                       ).pack(side='left')
        tk.Radiobutton(frm_input_type,
                       text='File is a list',
                       anchor='w',
                       value=Pipeline.INPUT_TYPE_LIST,
                       variable=self.input_type,
                       ).pack(side='left')
        self.input_type.set(Pipeline.INPUT_TYPE_ENTIRE)
        tk.Button(frm_input_type,
                  image=self.images['add'],
                  # text='  Add input file',
                  compound='left',
                  anchor='w',
                  command=self.AddInputFile,
                  ).pack(side='left')
        tk.Button(frm_input_type,
                  image=self.images['clear'],
                  # text='  Clear input file list',
                  compound='left',
                  anchor='w',
                  command=self.ClearInputFiles,
                  ).pack(side='left')

        frm_input_files = tk.Frame(frm_input)
        frm_input_files.pack(side='left', fill='x')

        input_file_columns = tuple(['menomonicname', 'inputtype'])
        self.input_files_tree = ttk.Treeview(frm_input_files,
                                             height=3,
                                             columns=input_file_columns)
        self.input_files_tree.pack(side='top',
                                   fill='both')
        self.input_files_tree.heading('#0', text='text')
        self.input_files_tree.heading('#1', text='mnemonic name')
        self.input_files_tree.heading('#2', text='input type')
        # Specify attributes of the columns (We want to stretch it!)
        self.input_files_tree.column('#0', minwidth=0, width=0, anchor='w',stretch=tk.NO)
        self.input_files_tree.column('#1', anchor='w', stretch=tk.YES)
        self.input_files_tree.column('#2', anchor='w', stretch=tk.YES)

    def _load_gptsettings(self):
        filename = Path(filedialog.askopenfilename(initialdir = os.getcwd(),
                                                   title = "Please select a file",
                                                   filetypes = (('json file', '*.json'),),
                                                   defaultextension='.json'
                                                   )
                        )
        if filename.name:
            #  read file
            self.selected_setting_filename.set(filename.name)
            self.gpt3settings = LoadJsonData(filename.absolute())
            #  add file ref
            self.gpt3settings['file'] = str(filename.absolute())

            try:
                #  set openai.key
                openai.api_key = self.gpt3settings["api-key"]
            except KeyError:
                messagebox.showerror('',
                                     'no valid API-key found!. Setting NOT loaded.')
                return

            self.pipeline.AddGPT3Settings(self.gpt3settings,)
            if self.gpt3settings:
                messagebox.showinfo('',
                                    'GPT-3 Settings loaded.')
            else:
                messagebox.showerror('',
                                     'Setting file corrupted.')

    def SelectInputFile(self):
        filename = Path(filedialog.askopenfilename(initialdir=os.getcwd(),
                                                   title="Please select a file",
                                                   filetypes=(('all files', '*.*'),
                                                              ('text file', '*.txt'),
                                                              ),
                                                   defaultextension='.txt'
                                                   )
                        )
        if filename.name:
            self.input_file_filename = str(filename.absolute())
            self.input_file_mnemonic_name.set(filename.name)

    def __create_frame_maintop_output(self):
        frm_output = tk.LabelFrame(self.__frm_maintop_left,
                                         text='Output',
                                         labelanchor='n')
        frm_output.pack(side='top',
                        fill='both',
                        expand=True)

        frm_output_script = tk.Frame(frm_output)
        frm_output_script.pack(side='top',
                               fill='x',
                               expand=True)

        tk.Button(frm_output_script,
                  # text='import module',
                  image=self.images['load'],
                  compound='center',
                  anchor='w',
                  height=self.BUTTON_HEIGHT,
                  command=self.LoadOutputFilterModuleGUI,
                  ).pack(side='left')
        self.cmb_pipeline_output_filter_name = ttk.Combobox(frm_output_script,
                                                            textvariable=self.pipeline_output_filter_name,
                                                            )
        self.cmb_pipeline_output_filter_name.pack(side='left',
                                                  fill='x',
                                                  expand=True)
        tk.Button(frm_output_script,
                  # text=' Add filter',
                  image=self.images['add'],
                  compound='center',
                  anchor='w',
                  height=self.BUTTON_HEIGHT,
                  command=self._add_output_script,
                  ).pack(side='left')
        tk.Button(frm_output_script,
                  # text=' Clear filters',
                  image=self.images['clear'],
                  compound='center',
                  anchor='w',
                  height=self.BUTTON_HEIGHT,
                  command=self._clear_output_script).pack(side='left')
        self.lst_output_script = tk.Listbox(frm_output,
                                            height=5, )
        self.lst_output_script.pack(side='top', fill='x')

    def _add_imported_output_filters_cmb(self, script_name):
        if type(self.cmb_pipeline_output_filter_name['values']) == str:
            self.cmb_pipeline_output_filter_name['values'] = [script_name]
        else:
            vals = list(self.cmb_pipeline_output_filter_name['values'])
            vals.append(script_name)
            vals = list(set(vals))
            self.cmb_pipeline_output_filter_name['values'] = vals
        self.pipeline_output_filter_name.set(self.cmb_pipeline_output_filter_name['values'][0])

    def _clear_item_output_filter(self, *event):
        self.lst_item_filters.delete(0, 'end')

    def _load_inputfile_content(self,
                                filename):
        if self.input_type.get() == Pipeline.INPUT_TYPE_ENTIRE:
            return codecs.open(filename, 'r', 'utf-8').read()
        else:
            # it is a list
            return [line.strip() for line in codecs.open(filename, 'r', 'utf-8').readlines() if line.strip()]

    def __create_frame_experimentitem(self):
        frm_experiment_item = tk.Frame(self.__frm_maintop_experimentitem)
        frm_experiment_item.pack(side='left',
                                 fill='both',
                                 expand=True)
        frm_explabel = tk.Frame(frm_experiment_item)
        frm_explabel.pack(side='top',
                          fill='x')
        tk.Label(frm_explabel,
                 text='Pipeline item name:'
                 ).pack(side='left')
        tk.Entry(frm_explabel,
                 textvariable=self.pipeline_item_name,
                 ).pack(side='left', fill='x', expand=True)


        frm_prompt = tk.Frame(frm_experiment_item)
        frm_prompt.pack(side='top', fill='x', expand=True)

        tk.Label(frm_prompt,
                 text='Prompt:',
                 anchor='w'
                 ).pack(side='left')
        self.cmb_prompt = ttk.Combobox(frm_prompt,
                                       text='Prompt List',
                                       textvariable=self.selected_prompt)
        self.cmb_prompt.pack(side='left',
                             fill='x',
                             expand=True)
        self.cmb_prompt.bind("<<ComboboxSelected>>", self._cmb_prompt_changed)
        self.cmb_prompt['state'] = 'readonly'

        tk.Button(frm_prompt,
                  # text='Show prompt',
                  image=self.images['show'],
                  compound='left',
                  # height=self.BUTTON_HEIGHT,
                  width=self.BUTTON_HEIGHT,
                  command=self._show_prompt,
                  ).pack(side='right')

        frm_key_val_filter = tk.Frame(frm_experiment_item)
        frm_key_val_filter.pack(side='top',
                  fill='x',
                  expand=True)

        frm_key_pair = tk.LabelFrame(frm_key_val_filter,
                                     text='Key-Value Bindings',
                                     labelanchor='n')
        frm_key_pair.pack(side='left',
                  fill='x',
                  expand=True)

        frm_expkey = tk.Frame(frm_key_pair)
        frm_expkey.pack(side='top', fill='x')
        tk.Label(frm_expkey,
                 text='prompt key',
                 ).pack(side='left', fill='x')
        self.cmb_input_prompt_key_name = ttk.Combobox(frm_expkey,
                                                      textvariable=self._prompt_key_name)
        self.cmb_input_prompt_key_name.pack(side='left', fill='x', expand=True)
        self.cmb_input_prompt_key_name['state'] = 'readonly'

        frm_expval = tk.Frame(frm_key_pair)
        frm_expval.pack(side='top', fill='x')
        tk.Label(frm_expval,
                 text='input value',
                 ).pack(side='left', fill='x')
        self.cmb_input_prompt_input_value = ttk.Combobox(frm_expval,
                                                         textvariable=self._prompt_input_value)
        self.cmb_input_prompt_input_value.pack(side='left', fill='x', expand=True)
        self.cmb_input_prompt_input_value['state'] = 'readonly'

        frm_buttons = tk.Frame(frm_key_pair)
        frm_buttons.pack(side='top', fill='x')

        tk.Button(frm_buttons,
                  text='Add key-value pair',
                  image=self.images['add'],
                  compound='left',
                  height=self.BUTTON_HEIGHT,
                  command=self._add_input_key_pair,
                  ).pack(side='left', fill='x', expand=True)

        tk.Button(frm_buttons,
                  text='delete pairs',
                  image=self.images['clear'],
                  compound='left',
                  height=self.BUTTON_HEIGHT,
                  command=self.ClearInputKeyValuePairs,
                  ).pack(side='left', fill='x', expand=True)


        frm_tree = tk.Frame(frm_key_pair)
        frm_tree.pack(side='top',
                      fill='both',
                      expand=True)
        input_key_pairs_columns = (['Prompt Key', 'type'])
        self.input_key_pairs = ttk.Treeview(frm_tree,
                                            columns=input_key_pairs_columns,
                                            height=5,
                                            )
        self.input_key_pairs.pack(side='left',
                                  fill='both',
                                  expand=True)
        # Set the heading (Attribute Names)
        self.input_key_pairs.heading('#0', text='text')
        self.input_key_pairs.heading('#1', text='Prompt Key')
        self.input_key_pairs.heading('#2', text='Input')
        # Specify attributes of the columns (We want to stretch it!)
        self.input_key_pairs.column('#0', minwidth=0, width=0, anchor='center',stretch=tk.NO)
        self.input_key_pairs.column('#1', anchor='center', stretch=tk.YES)
        self.input_key_pairs.column('#2', anchor='center', stretch=tk.YES)

        frm_filter = tk.LabelFrame(frm_key_val_filter,
                                   text='Item Output Filters',
                                   labelanchor='n')
        frm_filter.pack(side='right',
                  fill='x',
                  expand=True)

        frm_filter_buttons = tk.Frame(frm_filter)
        frm_filter_buttons.pack(side='top',
                        fill='x',
                        expand=True)
        tk.Button(frm_filter_buttons,
                  # text='  load filter module',
                  image=self.images['load'],
                  compound='left',
                  anchor='w',
                  command=self.LoadItemOutputModuleGUI,
                  ).pack(side='left', fill='x')
        self.cmb_pipeline_item_output_filter_name = ttk.Combobox(frm_filter_buttons,
                                                                 # text='select filter',
                                                                 textvariable=self.pipeline_item_output_filter_name,
                                                                 justify='left',
                                                                 )
        self.cmb_pipeline_item_output_filter_name.pack(side='left',
                                                       fill='x',
                                                       expand=True)
        self.cmb_pipeline_item_output_filter_name['state'] = 'readonly'

        frm_filter_type = tk.Frame(frm_filter)
        frm_filter_type.pack(side='top', fill='both')

        tk.Radiobutton(frm_filter_type,
                       text='on result',
                       variable=self.item_output_filter_type,
                       value=self.ITEM_OUTPUT_FILTER_TYPE_RESULTS,
                       anchor='w',
                       ).pack(side='left', fill='x')
        tk.Radiobutton(frm_filter_type,
                       text='ah-hoc on item',
                       variable=self.item_output_filter_type,
                       value=self.ITEM_OUTPUT_FILTER_TYPE_ITEM,
                       anchor='w',
                       ).pack(side='left', fill='x')
        #  set default value
        self.item_output_filter_type.set(self.ITEM_OUTPUT_FILTER_TYPE_RESULTS)

        frm_btns = tk.Frame(frm_filter)
        frm_btns.pack(side='top', fill='x')
        tk.Button(frm_btns,
                  text='  Add filter',
                  image=self.images['add'],
                  compound='left',
                  anchor='w',
                  command=self._add_item_output_filter,
                  ).pack(side='left', fill='x')
        tk.Button(frm_btns,
                  text='  Clear filters',
                  image=self.images['clear'],
                  compound='left',
                  anchor='w',
                  command=self._clear_item_output_filter,
                  ).pack(side='left', fill='x')

        self.lst_item_filters = tk.Listbox(frm_filter,
                                           # height=5,
                                           )
        self.lst_item_filters.pack(side='top', fill='both')

        tk.Button(frm_experiment_item,
                  text='   Add experimental item to pipeline',
                  anchor='center',
                  command=self.AddPipelineItemToPipeline,
                  image=self.images['add'],
                  compound='left',
                  height=self.BUTTON_HEIGHT,
                  ).pack(side='top',
                         fill='x',
                         expand=True)

    def _add_imported_item_output_filters_cmb(self, filter_name):
        if type(self.cmb_pipeline_item_output_filter_name['values']) == str:
            self.cmb_pipeline_item_output_filter_name['values'] = [filter_name]
        else:
            vals = list(self.cmb_pipeline_item_output_filter_name['values'])
            vals.append(filter_name)
            vals = list(set(vals))
            self.cmb_pipeline_item_output_filter_name['values'] = vals

        self.pipeline_item_output_filter_name.set(self.cmb_pipeline_item_output_filter_name['values'][0])

    def _fill_prompts_combobox(self):
        vals = list(self.pipeline.GetPromptNames())
        vals.insert(0, self.DO_NOTHING)
        self.cmb_prompt['values'] = vals

        if len(vals) > 1:
            self.selected_prompt.set(self.cmb_prompt['values'][1])
        else:
            self.selected_prompt.set(self.cmb_prompt['values'][0])
        self._cmb_prompt_changed()

    def _add_output_script(self, *event):
        if self.pipeline_output_filter_name.get():
            self.lst_output_script.insert('end', self.pipeline_output_filter_name.get())
            self.pipeline.AddOutputFilter(self.pipeline_output_filter_name.get())

    def _clear_imported_output_filters(self):
        self.cmb_pipeline_output_filter_name['values'] = []
        self.cmb_pipeline_output_filter_name.set('')
        self.pipeline.ClearOutputFilters()

    def _clear_imported_item_output_filters(self):
        self.cmb_pipeline_item_output_filter_name['values'] = []
        self.cmb_pipeline_item_output_filter_name.set('')
        # self.pipeline.ClearOutputFilters()

    def _add_input_key_pair(self, *event):
        if not self._prompt_key_name.get().strip() or not self._prompt_input_value.get().strip():
            return
        self.input_key_pairs.insert('',
                                  'end',
                                  text=self._prompt_key_name.get(),
                                  values=tuple([self._prompt_key_name.get(),
                                                self._prompt_input_value.get()]))
        #  remove used key from cmb list
        vals = list(self.cmb_input_prompt_key_name['values'])
        vals.remove(self._prompt_key_name.get())
        self.cmb_input_prompt_key_name['values'] = vals
        self._prompt_key_name.set('')

    def __create_frame_mainpipeline(self):
        #  macro container
        frm_pipeline = tk.LabelFrame(self.frm_main_pipeline,
                                     text='Pipeline',
                                     labelanchor='n')
        frm_pipeline.pack(side='top',
                         fill='both',
                         expand=True)

        pipeline_tree_columns = ('Experimantal Item', 'input',  'Prompt Name', 'Filter Name')
        self.pipeline_tree = ttk.Treeview(frm_pipeline,
                                          columns=pipeline_tree_columns
                                        )
        self.pipeline_tree.pack(side='top',
                                fill='both',
                                expand=True)
        self.pipeline_tree. bind('<Double-Button-1>', self.RemovePipelineItem)
        # Set the heading (Attribute Names)
        self.pipeline_tree.heading('#0', text='Step ID')
        self.pipeline_tree.heading('#1', text='Experimental Item')
        self.pipeline_tree.heading('#2', text='Prompt Name')
        self.pipeline_tree.heading('#3', text='Input')
        # self.pipeline_tree.heading('#4', text='Filter on Output')
        self.pipeline_tree.heading('#4', text='Filter list')

        # Specify attributes of the columns (We want to stretch it!)
        self.pipeline_tree.column('#0',
                                  anchor='center',
                                  width=50,
                                  minwidth=50,
                                  stretch=tk.NO)
        for col_n in range(1, len(pipeline_tree_columns)):
            col_id = '#{}'.format(col_n)
            self.pipeline_tree.column(col_id, anchor='center',
                                  minwidth=50,
                                  stretch=tk.YES)
