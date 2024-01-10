import sys
from copy import deepcopy
import tkinter as tk
import tkinter.ttk as ttk
import importlib
import importlib.machinery
import importlib.util
import os
import openai
from tkinter import messagebox
from tkinter import filedialog
from pathlib import Path
from promptdesignerdataset.dataset import PromptDesignerDataset, SaveJsonData
# from PIL import Image, ImageDraw, ImageTk, ImageFont
from itertools import product
from pipelinemaker.pipeline import PipelineItem
import pprint

from pipelinemaker.experimentmakerGUI import ExperimentGUI

class Experiment(ExperimentGUI):
    def __init__(self,
                 parent):
        self.parent = parent
        super(Experiment, self).__init__(self.parent)
        # super().__init__(self.parent)

    def NewPipeline(self, *event):
        self.ClearPipeline()

    def _pipeline_name_changed(self, *event):
        self.pipeline.SetName(self.pipeline_name.get())

    def LoadPromptDesignerDataset(self, *event):
        filename = Path(filedialog.askopenfilename(initialdir=os.getcwd(),
                                                   title="Please select a file",
                                                   filetypes=(('all files', '*.*'),
                                                              ('Prompt designer file', '*.answer-data'),
                                                              ),
                                                   defaultextension='.answer-data'
                                                   )
                        )
        if filename.name:
            experimental_base_dataset = PromptDesignerDataset()
            experimental_base_dataset.LoadData(str(filename.absolute()))
            for prompt_name in experimental_base_dataset.GetPromptNames():
                prompt_ = experimental_base_dataset.GetPrompt(prompt_name)
                self.pipeline.AddPrompt(prompt_name=prompt_name,
                                        prompt_template=prompt_)

            self._fill_prompts_combobox()

    def LunchExperimentalPipelineSelectedItem(self, *event):
        if not self.pipeline_tree.selection():
            return
        #  retrieve the item from pipeline tree
        item = self.pipeline_tree.selection()[0]
        values = self.pipeline_tree.item(item)['values']
        item_name = values[0]
        #  retrieve item from pipeline object
        item = self.pipeline.GetItem(item_name)
        # lunch experiment only for that item
        self.LunchItemExperiment(item)

    def LunchItemExperiment(self, item):
        def GetContent(v):
            if v in self.pipeline.GetInputFilesMnemonicNames():
                return self.pipeline.GetInputFileContent(v)
            else:
                prev = self.pipeline.GetItem(v)
                return prev.GetResultsContent()
        ############

        if not self._force_rerun_entire_pipeline.get():
            #  check if a result is present. if so, skip this item
            if item.GetResultsContent():
                return

        prompt_name = item.GetPromptName()
        if prompt_name == self.DO_NOTHING:
            #  raw results are the input of
            param = item.GetPromptParameters()
            v = list(param.values())[0]
            content__ = GetContent(v)
            if type(content__) == str:
                content__ = [content__]
            for content_item in content__:
                item.AddResultsRaw(content_item,
                                   param)
        else:
            prompt = self.pipeline.GetPrompt(prompt_name)
            prompt_parameters = item.GetPromptParameters()
            #  parameters substitution
            for k in list(prompt_parameters.keys()):
                v = prompt_parameters[k]
                prompt_parameters[k] = GetContent(v)
            # compose and collect all the prompt
            #  create all the possible prompts
            prompts_to_ask = list()

            # for each non list parameter, I cast it in a list
            for k, v in prompt_parameters.items():
                if type(v) != list:
                    prompt_parameters[k] = [v]
            keys, values = zip(*prompt_parameters.items())
            permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
            print('combinations:', len(permutations_dicts), permutations_dicts)
            # print('unique combinations: ', len(set(permutations_dicts)))
            #  create prompts
            for combination in permutations_dicts:
                prompt_to_ask = {'composed-prompt': prompt.format(**combination),
                                 'key-value-pairs': combination}
                prompts_to_ask.append(prompt_to_ask)
            #  ask prompts and collect results
            results = list()
            for prompt_to_ask in prompts_to_ask:
                # print('fix results - remove DEV mode!')
                result = AskGPT3(prompt_to_ask['composed-prompt'],
                                 self.gpt3settings)
                # results.append(result)
                item.AddResultsRaw(result,
                                   prompt_to_ask['key-value-pairs'])

        for filter_name, filter_type in item.GetItemFilterNames():
            # print(filter_name, filter_type)
            if filter_type == self.ITEM_OUTPUT_FILTER_TYPE_RESULTS:
                _res_item = item.GetResultsContent()
                results_filtered = self.imported_item_output_filtes[filter_name](_res_item)
            else:
                #  apply an ad-hoc filter on the item.
                #  this means that the filter might consider multiple items
                #  So, I pass the entire pipeline data
                data = self.pipeline.PrepareDataToSave()
                results_filtered = self.imported_item_output_filtes[filter_name](data)

            item.AddResultsFiltered(results_filtered,
                                    filter_name)

    def LunchExperimentalPipeline(self, *event):
        #  lunch experiments

                #  it is of type list of len 1
                # content = content[0]
        if self._force_rerun_entire_pipeline.get():
            #  avoid adding multiple times the same results
            self.pipeline.ClearResults()

        for item in self.pipeline.GetItems():
            self.LunchItemExperiment(item)

        #  show results
        for item in self.pipeline.GetItems():
            print(item.Name(), item.GetStep())
            print(item.GetResultsContent())
        #  end of experiment
        #  apply outputscript if present
        if self.pipeline.GetOutputFilters():
            self.LunchOutputScript()
        messagebox.showinfo('','Experiment Completed.')

    def LunchOutputScript(self, *event):
        # global_results = list()
        results_pipeline = self.pipeline.ExportResults()
        for filter_name in self.pipeline.GetOutputFilters():
            results = self.imported_output_filters[filter_name](results_pipeline)
            print(filter_name)
            print(results)
        messagebox.showinfo('', 'output script applied')

    def SaveAsPipeline(self, *event):
        filename = Path(filedialog.asksaveasfilename(initialdir=os.getcwd(),
                                                     title="Please select a file",
                                                     filetypes=(('Pipeline file', '*.pipeline'),),
                                                     defaultextension='.pipeline',
                                                     )
                        )
        if filename.name:
            return str(filename.absolute())

    def _show_gpt3_settings(self):
        def close():
            app.destroy()
        app = tk.Toplevel(self.parent)
        app.title(self.selected_setting_filename.get())
        frm = tk.Frame(app)
        frm.pack(side='top', fill='both', expand=True)
        t = ttk.Treeview(frm, columns=(['key', 'value']))
        t.heading('#0', text='#')
        t.heading('#1', text='Key')
        t.heading('#2', text='Value')
        t.column('#0',
                anchor='w',
                width=30,
                minwidth=20,
                stretch=tk.NO)
        t.column('#1', anchor='w', stretch=tk.YES)
        t.column('#2', anchor='w', stretch=tk.YES)
        t.pack(side='left',
               fill='both',
               expand=True)
        ys = ttk.Scrollbar(frm,
                           orient='vertical',
                           command=t.yview)
        t['yscrollcommand'] = ys.set
        ys.pack(side='right', fill='y')
        tk.Button(app,
                  text='Close',
                  anchor='center',
                  command=close).pack(side='top',
                                      fill='x')
        # insert promptdesignerdataset
        for n_, item in enumerate(self.gpt3settings.items()):
            k, v = item

            if type(v) == list:
                v = ' '.join(v).strip()
            t.insert('', n_, text=str(n_), values=(k,v))
            #
            # row = '{key:<25}: {value:<25}'.format(key=k,
            #                                       value=v)
            # t.insert(n_, row)
            #
            # index = '{}.0'.format(n_+1)
            # t.insert(index,
            #          row)
        app.mainloop()

    def InputFilePreview(self):
        #  if no file selected, exit the method
        if not self.input_file_filename:
            return

        def close():
            app.destroy()
        app = tk.Toplevel(self.parent)
        app.title(str(Path(self.input_file_filename).name))
        frm = tk.Frame(app)
        frm.pack(side='top', fill='both', expand=True)
        t = tk.StringVar()
        tk.Label(frm,
                 textvariable=t,
                 anchor='w',
                 justify='left',
                 background='blue',
                 foreground='white',
                 ).pack(side='left',
                        fill='both',
                        expand=True)
        tk.Button(app,
                  text='Close',
                  anchor='center',
                  command=close).pack(side='top',
                                      fill='x')
        # insert promptdesignerdataset
        content = self._load_inputfile_content(self.input_file_filename)
        if type(content) == str:
            t.set(content)
        else:
            t.set('\n'.join(content).strip())
        app.mainloop()

    def _show_prompt(self):
        def close():
            app.destroy()
        app = tk.Toplevel(self.parent)
        app.title(self.selected_prompt.get())
        frm = tk.Frame(app)
        frm.pack(side='top', fill='both', expand=True)
        t = tk.Text(frm,
                    height=23,
                    width=180,
                    wrap='word', )
        t.pack(side='left',
               fill='both',
               expand=True)
        ys = ttk.Scrollbar(frm,
                           orient='vertical',
                           command=t.yview)
        t['yscrollcommand'] = ys.set
        ys.pack(side='right', fill='y')
        tk.Button(app,
                  text='Close',
                  anchor='center',
                  command=close).pack(side='top',
                                      fill='x')
        # insert promptdesignerdataset
        if self.selected_prompt.get().strip():
            t.insert('1.0', self.pipeline.GetPrompt(self.selected_prompt.get()))
            app.mainloop()

    def LoadItemOutputModuleGUI(self, *event):
        outputscript_file = Path(filedialog.askopenfilename(initialdir=os.getcwd(),
                                                            title="Please select a file",
                                                            filetypes=(('all files', '*.*'),
                                                                          ('Python file', '*.py'),
                                                                          ),
                                                            defaultextension='.py',
                                                            )
                                 )
        if outputscript_file.name:
            self.LoadItemOutputModule(str(outputscript_file.absolute()))
            #  add ref to pipeline object
            self.pipeline.AddItemOutputScriptModule(str(outputscript_file.absolute()))

    def LoadItemOutputModule(self,
                             scriptfile):
        try:
            scriptfile = Path(scriptfile)
            # get name
            mnemonic_outputscript_name = scriptfile.name
            # Import mymodule
            loader = importlib.machinery.SourceFileLoader(mnemonic_outputscript_name,
                                                          str(scriptfile.absolute()))
            spec = importlib.util.spec_from_loader(mnemonic_outputscript_name, loader)
            importedscript = importlib.util.module_from_spec(spec)
            loader.exec_module(importedscript)
            #  filter out class elements that has no a Parse Method
            namespaces = [i for i in dir(importedscript) if not i.startswith('__')]
            # imported_filters = dict()
            for namespace in namespaces:
                if type(importedscript.__dict__[namespace]) == type:
                    # it is a class
                    namespace_object = importedscript.__dict__[namespace]().Parse
                else:
                    # it is a function
                    namespace_object = importedscript.__dict__[namespace]
                # if the imported function/module pass the test it is added
                try:
                    namespace_object('test string')
                    namespace_object(['test list 1', 'test list 2'])
                    self.imported_item_output_filtes [namespace] = namespace_object
                    self._add_imported_item_output_filters_cmb(namespace)
                except TypeError:
                    continue
                except NotImplementedError:
                    continue
        except:
            messagebox.showerror('',
                                 'The module {} is corrupted or missing. please re-load it.'.format(scriptfile))

    def LoadOutputFilterModuleGUI(self, *event):
        outputscript_file = Path(filedialog.askopenfilename(initialdir=os.getcwd(),
                                                            title="Please select a file",
                                                            filetypes=(('all files', '*.*'),
                                                                       ('Python file', '*.py'),
                                                                       ),
                                                            defaultextension='.py',
                                                            )
                                 )
        if outputscript_file.name:
            self.LoadOutputFilterModule(str(outputscript_file.absolute()))
            #  add ref to pipeline object
            self.pipeline.AddOutputScriptModule(str(outputscript_file.absolute()))

    def LoadOutputFilterModule(self,
                                  filename):
        try:
            # get name
            outputscript_file = Path(filename)
            mnemonic_outputscript_name = outputscript_file.name
            # Import mymodule
            loader = importlib.machinery.SourceFileLoader(mnemonic_outputscript_name, str(outputscript_file.absolute()))
            spec = importlib.util.spec_from_loader(mnemonic_outputscript_name, loader)
            importedscript = importlib.util.module_from_spec(spec)
            loader.exec_module(importedscript)
            #  filter out class elements that has no a Parse Method
            namespaces = [i for i in dir(importedscript) if not i.startswith('__')]
            # imported_filters = dict()
            for namespace in namespaces:
                if type(importedscript.__dict__[namespace]) == type:
                    # it is a class
                    try:
                        namespace_object = importedscript.__dict__[namespace]().Parse
                    except AttributeError:
                        continue
                    except TypeError:
                        continue
                else:
                    continue
                    # No function allowed
                    # # it is a function
                    # namespace_object = importedscript.__dict__[namespace]
                # if the imported function/module pass the test it is added
                try:
                    # namespace_object('test string')
                    self.imported_output_filters[namespace] = namespace_object
                    self._add_imported_output_filters_cmb(namespace)

                except TypeError:
                    continue
                except NotImplementedError:
                    continue
        except:
            messagebox.showerror('',
                                 'The module {} is corrupted or missing. please re-load it.'.format(filename))

    def ExportResultsGUI(self):
        filename = Path(filedialog.asksaveasfilename(initialdir=os.getcwd(),
                                                     title="Please select a file",
                                                     filetypes=(('JSON file', '*.json'),),
                                                     defaultextension='.json',
                                                     )
                        )
        if filename.name:
            exported_data = self.pipeline.ExportResults()
            if SaveJsonData(exported_data, str(filename.absolute())):
                messagebox.showinfo('',
                                    'Results Exported.')

    def ShowPipelineItem(self):
        def close():
            app.destroy()

        if not self.pipeline_tree.selection():
            return
        #  retrieve the item from pipeline tree
        item = self.pipeline_tree.selection()[0]
        values = self.pipeline_tree.item(item)['values']
        item_name = values[0]
        #  retrieve item from pipeline object
        item = self.pipeline.GetItem(item_name)

        item_val_dict = item.GetItemDict()

        #  create GUI
        app = tk.Toplevel(self.parent)
        app.title('Pipeline Item: {}'.format(item_val_dict['name']))
        app.geometry("350x150")
        frm = tk.Frame(app)
        frm.pack(side='top', fill='both', expand=True)
        tree = ttk.Treeview(frm, columns=('val'))
        tree.pack(fill='both', expand=True)
        tree.heading('#0', text='text')
        tree.heading('#1', text='value')
        # tree.heading('#2', text='input type')
        # Specify attributes of the columns (We want to stretch it!)
        tree.column('#0', minwidth=50, width=150, anchor='w', stretch=tk.NO)
        tree.column('#1', width=250, anchor='w', stretch=tk.YES)

        tree.insert('',
                      'end',
                      text='Step',
                      values=(str(item.GetStep())))
        val = '"{}"'.format(str(item_val_dict['prompt']))
        tree.insert('',
                      'end',
                      text='prompt name',
                      values=(val))
        print(item_val_dict['prompt'])
        val = ('')
        tree.insert('',
                      'end',
                      iid='input_item',
                      text='input items',
                      values=val)
        tree.insert('input_item',
                      'end',
                      iid='input-bindings',
                      text='input-bindings',
                      values=val)
        tree.insert('input_item',
                      'end',
                      iid='apply-item-filters',
                      text='apply-item-filters',
                      values=val)

        for n_f, filter in enumerate(item_val_dict['input_item']['apply-item-filters']):
            str_filter = '"{} - {}"'.format(*filter)
            tree.insert('apply-item-filters',
                          'end',
                          text='item filter {}'.format(n_f),
                          values=(str_filter)
                        )


        for n_b, binding in enumerate(item_val_dict['input_item']['input-bindings']):
            tree.insert('input-bindings',
                          'end',
                          text='bindings {}'.format(n_b),
                          values=(str(binding)))

    def ShowResults(self):
        #  collect Exproted promptdesignerdataset
        exported_data = self.pipeline.ExportResults()
        #  show promptdesignerdataset
        pp = pprint.PrettyPrinter()
        exported_data_str = pp.pformat(exported_data)
        # print(exported_data_str)

        def close():
            app.destroy()
        app = tk.Toplevel(self.parent)
        app.title('Results')
        frm = tk.Frame(app)
        frm.pack(side='top', fill='both', expand=True)
        t = tk.Listbox(frm)
        t.pack(side='left', fill='both', expand=True)

        for line in exported_data_str.split('\n'):
            t.insert('end', line)
        ys = ttk.Scrollbar(frm,
                           orient='vertical',
                           command=t.yview)
        t['yscrollcommand'] = ys.set
        ys.pack(side='right', fill='y')
        tk.Button(app,
                  text='Close',
                  anchor='center',
                  command=close).pack(side='top',
                                      fill='x')
        app.mainloop()

    ### THIS SHOULD BE MOVED IN prompt_designer_dataset
    def get_prompt_parameters(self,
                              prompt_name: str):

        prompt = self.pipeline.GetPrompt(prompt_name)

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
                if found_ind > 0:
                    begin_index = found_ind + 1
                    # move index
                    index = found_ind + 1
                    look_for_begin = False
                else:
                    break
            else:
                found_ind = prompt.find('}', index)
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

        return list(set(parameter_names))

    def _cmb_prompt_changed(self, *event):
        if self.selected_prompt.get() == self.DO_NOTHING:
            self.cmb_input_prompt_key_name['values'] = ['input']
        else:
            keys = self.get_prompt_parameters(self.selected_prompt.get())
            self.cmb_input_prompt_key_name['values'] = keys
        self._clear_input_key_pairs()

            #  restore parameter keys
        # self._cmb_prompt_changed()
    def ClearInputKeyValuePairs(self, *event):
        self._clear_input_key_pairs()
        self._cmb_prompt_changed()

    def ClearPipeline(self, internal_callback=False):
        #  internal callback suppress warning
        if not internal_callback:
            answer = messagebox.askyesno('', 'Are you sure you want to delete the pipeline?')
            if answer == tk.NO:
                return
        #  clear pipeline
        self.pipeline.Clear()
        self.ClearInputKeyValuePairs()
        self._clear_input_values()

        #  clear pipeline tree
        for item in self.pipeline_tree.get_children():
            self.pipeline_tree.delete(item)

    def RemovePipelineItem(self, event):
        if not event.widget.selection():
            return
        if messagebox.askyesno('',
                               'remove item?') == tk.YES:
            item = event.widget.selection()
            values = self.pipeline_tree.item(item)['values']
            item_name = values[0]
            self.pipeline.RemoveItem(item_name=item_name)
            self.pipeline_tree.delete(item)
            print('item ', item, 'removed')
            #  remove from combobox
            cmb_vals = list(self.cmb_input_prompt_input_value['values'])
            cmb_vals.remove(item_name)
            self.cmb_input_prompt_input_value['values'] = cmb_vals

    def AddInputFile(self):
        file_ = self.input_file_filename
        if not file_:
            messagebox.showerror('',
                                 'You must select a file.')
            return
        filename = str(Path(file_).name)
        mnemonic_name = self.input_file_mnemonic_name.get()
        if not mnemonic_name:
            messagebox.showerror('',
                                 'You must set a mnemonic name for this file')
            return
        inputtype = self.input_type.get()
        content = self._load_inputfile_content(file_)

        inputfileitem = {'file': file_,
                         'input-type': inputtype,
                         'content': content,
                         'mnemonic-name': mnemonic_name}

        self.pipeline.AddInputFile(inputfileitem)
        self._add_inputfile_into_filetreelist(filename, mnemonic_name, inputtype)


    def ClearInputFiles(self):
        #  clean tree view widget
        for item in self.input_files_tree.get_children():
            self.input_files_tree.delete(item)
        #  clear combobox widget
        self.cmb_input_prompt_input_value['values'] = []
        self.cmb_input_prompt_input_value.set('')

    def LoadPipeline(self):
        filename = Path(filedialog.askopenfilename(initialdir=os.getcwd(),
                                                   title="Please select a file",
                                                   filetypes=(('all files', '*.*'),
                                                              ('Pipeline file', '*.pipeline'),
                                                              ),
                                                   defaultextension='.pipeline',
                                                   )
                        )
        if filename.name:

            self.pipeline.Clear()
            #  clear input file
            self.ClearInputFiles()
            #  clear previous pipeline
            self.ClearPipeline(internal_callback=True)
            #  clear output script list
            self._clear_output_script()

            #  load pipeline
            self.pipeline.LoadPipelilne(str(filename.absolute()))
            self.output_pipeline_filename.set(self.pipeline.GetFile())

            #  load promptdesignerdataset into GUI
            self.pipeline_name.set(self.pipeline.GetName())

            for fileitem in self.pipeline.GetInputFilesItems():
                self._add_inputfile_into_filetreelist(filename_name=str(Path(fileitem['file']).name),
                                                      mnemonic_name=fileitem['mnemonic-name'],
                                                      inputtype=fileitem['input-type'],)

            self.gpt3settings = self.pipeline.GetGPT3Settings()
            try:
                openai.api_key = self.gpt3settings["api-key"]
            except:
                print('No settings found! no api-key set.')

            #  load pipeline items
            for item in self.pipeline.GetItems():
                #  add item to pipeline tree view
                self.pipeline_tree.insert('',
                                          'end',
                                          text=str(item.GetStep()),
                                          values=item.GetTreeItem())
                #  add item name to input list
                self._add_input_type_item(item.Name())

            self.pipeline_item_name.set('')
            #  fix combo  experiment item
            self.ClearInputKeyValuePairs()

            #  load modules
            for mod in self.pipeline.GetItemOutputScriptModules():
                self.LoadItemOutputModule(mod)

            for mod in self.pipeline.GetOutputScriptModules():
                self.LoadOutputFilterModule(mod)

            #  load output filters
            for filter in self.pipeline.GetOutputFilters():
                self.lst_output_script.insert('end', filter)
            messagebox.showinfo('',
                        'Pipeline Loaded.')

            self._fill_prompts_combobox()



    def SavePipeline(self):
        if not self.pipeline_name.get().strip():
            messagebox.showwarning('', 'You must set a name for the pipeline.')
            return
        if not self.output_pipeline_filename.get():
            filename = self.SaveAsPipeline()
            if not filename:
                return
            self.output_pipeline_filename.set(filename)
        self.pipeline.SavePipeline(self.output_pipeline_filename.get())
        messagebox.showinfo('', 'Experimental pipeline saved.')

    def AddPipelineItemToPipeline(self):
        if not self.pipeline_item_name.get().strip():
            messagebox.showinfo('',
                                'You must set a name for this pipeline item.')
            return

        if self.pipeline_item_name.get() in list(self.cmb_input_prompt_input_value['values']):
            messagebox.showinfo('',
                                'Pipeline Name already in used, please set a new one.')
            return

        if len(list([v for v in self.cmb_input_prompt_key_name['values'] if v])) > 0:

            messagebox.showerror('',
                                 'You must set each prompt key name before adding this item to the pipeline.')
            return

        #  create pipeline item
        item_input = self.__create_input_item()
        item = PipelineItem()
        item.SetName(self.pipeline_item_name.get())
        item.SetPrompt(self.selected_prompt.get())
        item.SetInput(input_item=item_input)

        self.pipeline.AddItem(deepcopy(item))
        #  add item to pipeline tree view
        self.pipeline_tree.insert('',
                                  'end',
                                  text=str(self.pipeline.total_steps()),
                                  values=item.GetTreeItem())
        #  add item name to input list
        self._add_input_type_item(item.Name())
        self.pipeline_item_name.set('')
        #  fix combo  experiment item
        self.ClearInputKeyValuePairs()
        self._clear_item_output_filter()

    def __create_input_item(self):
        item_input = {'apply-item-filters': [i for i in self.lst_item_filters.get(0, 'end')], #self.chk_apply_filter_to_pipeline_item_output.get(),
                      'input-bindings': list(),
                      }
        for child in self.input_key_pairs.get_children():
            child = self.input_key_pairs.item(child)['values']
            item_input['input-bindings'].append({'prompt-key': child[0],
                                                 'key-value': child[1]})
        return deepcopy(item_input)

######
######
#####


def AskGPT3(prompt,
            gpt_settings):

    # def _extract_answers(answers):
    #     return [ans['text'] for ans in answers['choices']]
    #
    # print('GPT3 Dev Mode')
    # return "this is a list\n of results\n \n   A:   \n -aaaa\n -'b'"
    try:
        response = openai.Completion.create(
                            engine=gpt_settings[PromptDesignerDataset.ENGINE],
                            prompt=prompt,
                            max_tokens=int(gpt_settings[PromptDesignerDataset.MAX_TOKEN]),
                            stop=gpt_settings[PromptDesignerDataset.STOPWORDS_LIST],
                            temperature=gpt_settings[PromptDesignerDataset.TEMPERATURE],
                            top_p=gpt_settings[PromptDesignerDataset.NUCLEUS],
                            n=gpt_settings[PromptDesignerDataset.N_],
                            presence_penalty=gpt_settings[PromptDesignerDataset.PRESENCE_PENALTY],
                            frequency_penalty=gpt_settings[PromptDesignerDataset.FREQUENCY_PENALTY],
                            )
        return response['choices'][0]['text']

    except Exception as err:
        messagebox.showerror('',
                             'OpenAI GPT-3 Error: {}'.format(str(err))
                             )
        return err

def LunchExperiment():

    app = tk.Tk()
    Experiment(app)
    app.mainloop()

    sys.exit(0)

if __name__ == '__main__':
    LunchExperiment()