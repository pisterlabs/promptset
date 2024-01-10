# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:20:20 2023

@author: ManuMan
"""
import sys
import os
import json
from PIL import Image,ImageTk, ImageSequence
import subprocess as sp

import customtkinter as ctk
from customtkinter import CTkTextbox, CTkImage, CTkFont, filedialog
from tktooltip import ToolTip

from cortana.api.encrypt import encrypt_key
from cortana.model.utils import create_icon, load_markdown_file
from cortana.model.external_classes import MarkdownOutput

###############################################################################
# Dropdown menu
###############################################################################
directory = os.path.dirname(__file__)
with open(os.path.join(directory, 'parameters.json')) as json_file:
    kwargs = json.load(json_file)
    
markdown = kwargs['app']['markdown']

def create_dropdown_model(self, landing_page=False):
    
    if landing_page:

        # Dropdown menu options
        options = [
            "gpt-4",
            "gpt-4-32k",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
          
        # Create Dropdown menu
        self.label_model = ctk.CTkLabel(self, text="Model", text_color='cyan', bg_color="#13273a", height=18)
        self.drop_model = ctk.CTkComboBox(self, state="readonly", values=options, bg_color='#13273a', command=self.change_model) 
        ToolTip(self.drop_model, msg="Select the LLM model you want to use. 16k and 32k indicate the number of token in memory.", delay=1.0)

        self.drop_model.set(options[0])
        self.change_model(options[0], False)
        self.label_model.grid(row = 8, column = 0, padx=(0, 100), pady=(70, 5))
        self.drop_model.grid(row = 8, column = 0, padx=(0, 0), pady=(130, 10))
    else:
        self.label_model.grid(row = 8, column = 0, padx=(0, 100), pady=(10, 10))
        self.drop_model.grid(row = 8, column = 0, padx=(0, 0), pady=(60, 10))

def create_dropdown_role(self, landing_page=False):
    # Dropdown menu options
    if landing_page:

        with open(self.prepromt_path) as json_file:
            kwargs = json.load(json_file)
        roles = kwargs['roles']
        options = [role for role in roles.keys()]
        
        # Create Dropdown menu
        self.label_role = ctk.CTkLabel(self, text="Role", text_color='cyan', bg_color="#13273a", height=18)
        self.drop_role = ctk.CTkComboBox(self, state="readonly", values=options, bg_color='#13273a', command=self.change_role) 
        ToolTip(self.drop_role, msg="Select the role you want the AI to play. Check the preprompt button to know what the roles are.\n"\
                                    "You can add any role you want in the json, it will automatically add the role name in the list", delay=1.0)

        self.drop_role.set(options[0])
        self.change_role(options[0], False)
        self.label_role.grid(row = 8, column = 1, padx=(0, 100), pady=(70, 5))
        self.drop_role.grid(row = 8, column = 1, padx=(0, 0), pady=(130, 10))
    else:
        self.label_role.grid(row = 8, column = 1, padx=(0, 100), pady=(10, 10))
        self.drop_role.grid(row = 8, column = 1, padx=(0, 0), pady=(60, 10))
        

###############################################################################
# Launching functions
###############################################################################

def launch_language(self):
    button_language(self)
    self.lang_label = ctk.CTkLabel(self, text="Select language!", text_color='white', bg_color="#19344d")
    self.lang_label.grid(row = 0, column = 0, padx=(5, 10), pady=(150, 0))
    
def launch_parameter(self):
    button_parameter(self)
    self.param_label = ctk.CTkLabel(self, text="Provide path to markdown editor!", text_color='white')
    self.param_label.grid(row = 8, column = 4, padx=(0, 0), pady=(130, 10))
    
def regular_app_layout(self):
    button_language(self)
    regular_app_buttons(self)

###############################################################################
# Button functions
###############################################################################


def button_language(self):
    self.button_fr = ctk.CTkButton(self, text="En",background_corner_colors=['#13273a', '#13273a', '#485a69', '#13273a'],
                                   width=70,
                                   command=lambda:self.launch_cortana('english', api_key=self.get_api_key(), role=self.role, **kwargs))
    self.button_en = ctk.CTkButton(self, text="Fr", background_corner_colors=['#13273a', '#485a69', '#485a69', '#3e4d5a'],
                                   width=70, 
                                   command=lambda:self.launch_cortana('french', api_key=self.get_api_key(), role=self.role, **kwargs))
    self.button_fr.grid(row = 0, column = 0, padx=(5, 10), pady=(10, 0))
    self.button_en.grid(row = 0, column = 0, padx=(5, 10), pady=(80, 0))

def button_parameter(self):
    if not hasattr(self, "img_param"):
        self.img_param = create_icon(self, "param.png")
    self.button_param = ctk.CTkButton(self, image=self.img_param.image, width=10, height=10,
                                      border_width=0, fg_color="#13273a", bg_color="#13273a",# background="#13273a",
                                  command=lambda:sp.Popen([self.open_param, self.param_path]))
    self.button_param.grid(row = 8, column = 5, padx=(0, 0), pady=(130, 10))
    ToolTip(self.button_param, msg="Open the parameter json file. The first you launch the app you must provide the markdown editor path.", delay=1.0)

def checkbox_reload(self):
    self.check_autoreload = ctk.CTkCheckBox(self, variable=self.autoreload_var, command=lambda:checkbox_function(self),
                                            text='Auto reload', onvalue=1, offvalue=0, bg_color="#13273a", text_color='cyan')
    ToolTip(self.check_autoreload, msg="Automatically close and reopen the markdown file", delay=1.0)   # True by default

def checkbox_function(self):
    self.autoreload = self.autoreload_var.get()

def create_button_api(self):
    if not hasattr(self, "img_api"):
        self.img_api = create_icon(self, "api.png")
    
    self.button_api = ctk.CTkButton(self, image=self.img_api.image, 
                                    bg_color="#13273a",
                                    width=50, height=20,#borderwidth=0, pady=0, padx=0, background="#13273a",
                                    command=lambda:create_apikey(self))
    self.button_api.grid(row = 7, column = 0, padx=(100, 0), pady=(200, 0))
    

def load_from_file(self):
    filename = filedialog.askopenfilename(initialdir=self.folder_res)
    sp.Popen([markdown, filename])
    last_messages = load_markdown_file(filename)
    self.filename = os.path.basename(filename).split('.md')[0]
    self.my_cortana.messages.append({'role':'user', "content":last_messages})
    self.markdown_output = MarkdownOutput(self.filename, last_messages)
    sys.stdout = self.markdown_output


def regular_app_buttons(self):
    
    self.button_param.grid_remove()
    self.drop_model.grid_remove()
    self.drop_role.grid_remove()
    self.label_model.grid_remove()
    self.label_role.grid_remove()
    create_dropdown_model(self)
    create_dropdown_role(self)
    
    if self.lang_label is not None:
        self.lang_label.grid_remove()
        self.param_label.grid_remove()
    else:
        self.lang_label = None
        self.param_label = None
        
    if not hasattr(self, "img_talk"):
        self.img_talk = create_icon(self, "icon1.png")
    
    if not hasattr(self, "img_open"):
        self.img_open = create_icon(self, "icon-open.png")
    
    if not hasattr(self, "img_folder"):
        self.img_folder = create_icon(self, "folder.png")

    if not hasattr(self, "img_plus"):
        self.img_plus = create_icon(self, "plus.png")

    if not hasattr(self, "img_load"):
        self.img_load = create_icon(self, "load.png")
        
    if not hasattr(self, "img_preprompt"):
        self.img_preprompt = create_icon(self, "preprompt.png")
        
    self.button_enter = ctk.CTkButton(self, text="Enter", 
                                      command=lambda:self.start_chat(self.entry_prompt.get("1.0" , ctk.END)), )
    self.button_talk = ctk.CTkButton(self, image=self.img_talk.image, border_width=0, 
                                     bg_color="#13273a", fg_color="#13273a",
                                     width=20, height=20,
                                     command=lambda:active_mode(self))
    ToolTip(self.button_talk, msg="Launch the active conversation mode, have fun!", delay=1.0)
    
    self.button_file = ctk.CTkButton(self, image=self.img_open.image, border_width=0 , 
                                     background_corner_colors=['#388ab0', '#13273a', '#13273a', '#13273a'],
                                     width=50, height=20, #borderwidth=0, pady=0, padx=0, background="white",
                                     command=lambda:sp.Popen([markdown, os.path.join(self.folder_res, f'{self.filename}.md')]))
    ToolTip(self.button_file, msg="Open the current markdown file in conversation", delay=1.0)
    
    self.button_folder = ctk.CTkButton(self, image=self.img_folder.image, border_width=0, 
                                    bg_color="#13273a", fg_color="#13273a",
                                    width=50, height=20,
                                    command=lambda:os.startfile(self.folder_res))
    ToolTip(self.button_folder, msg="Open the folder where markdown files are stored", delay=1.0)
    
    self.button_new = ctk.CTkButton(self, image=self.img_plus.image, border_width=0, 
                                    bg_color="#13273a", fg_color="#13273a",
                                    width=20, height=20,
                                    command=lambda:self.new_filename())
    ToolTip(self.button_new, msg="Open conversation in a new file", delay=1.0)
    
    self.button_load = ctk.CTkButton(self, image=self.img_load.image, 
                                     background_corner_colors=['#13273a', '#13273a', '#13273a', '#13273a'],
                                     width=50, height=20,#borderwidth=0, pady=0, padx=0, background="#13273a",
                                     command=lambda:load_from_file(self))
    ToolTip(self.button_load, msg="Load an existing markdown file", delay=1.0)

    self.button_preprompt = ctk.CTkButton(self, image=self.img_preprompt.image, bg_color="#13273a",
                                          width=50, height=20,#borderwidth=0, pady=0, padx=0, background="#13273a",
                                          command=lambda:sp.Popen([self.open_param, self.prepromt_path]))
    ToolTip(self.button_preprompt, msg="Open the preprompt parameter to customize your Cortana", delay=1.0)

    self.button_param = ctk.CTkButton(self, image=self.img_param.image, width=10, height=10,
                                      border_width=0, fg_color="#13273a", bg_color="#13273a",# background="#13273a",
                                  command=lambda:sp.Popen([self.open_param, self.param_path]))
    ToolTip(self.button_param, msg="Parameters of the app, open the json file", delay=1.0)   # True by default
    
    checkbox_reload(self)
    
    self.button_talk.grid(row = 8, column = 9, padx=(0, 0), pady=(60, 10))
    self.button_folder.grid(row = 8, column = 8, padx=(0, 0), pady=(60, 10))
    self.button_new.grid(row = 8, column = 7, padx=(0, 0), pady=(60, 10))
    self.button_param.grid(row = 8, column = 5, padx=(0, 0), pady=(60, 10))
    
    self.button_file.grid(row = 7, column = 10, padx=(50, 0), pady=(100, 10))
    self.button_load.grid(row = 7, column = 10, padx=(50, 0), pady=(180, 0))
    self.check_autoreload.grid(row = 8, column = 10, padx=(10, 10), pady=(5, 0))
    
    self.button_preprompt.grid(row = 7, column = 0, padx=(0, 10), pady=(200, 0))
    create_button_api(self)

    self.button_enter.grid(row = 10, column = 0, columnspan=11, sticky='sew', padx=(10, 10), pady=(0, 10))

def remove_buttons(self):
    self.button_enter.grid_remove()
    self.drop_model.grid_remove()
    self.label_model.grid_remove()
    self.button_talk.grid_remove()
    self.button_preprompt.grid_remove()
    self.check_autoreload.grid_remove()
    self.button_param.grid_remove()
    self.drop_model.grid_remove()
    self.drop_role.grid_remove()
    self.label_model.grid_remove()
    self.label_role.grid_remove()
    self.button_api.grid_remove()
    self.button_fr.grid_remove()
    self.button_en.grid_remove()
    self.button_load.grid_remove()
    self.button_new.grid_remove()
    self.button_file.grid_remove()
    self.button_folder.grid_remove()

def create_apikey(self):
    remove_buttons(self)
    self.api_label = ctk.CTkLabel(self, text="Enter API key from OpenAi", text_color='cyan', bg_color="#13273a")
    self.api_label.grid(row = 8, column = 0, padx=(20, 0), pady=(5, 5))
    self.api_entry = ctk.CTkEntry(self, bg_color="#13273a", width=300)
    self.api_entry.grid(row = 8, column = 0, padx=(20, 0), pady=(70, 10))#ipadx=40, padx=35, pady=160, anchor='sw')
    self.button_ok = ctk.CTkButton(self, text="OK", command=lambda:remove_api(self), width=10, height=10)
    self.button_ok.grid(row = 8, column = 1, padx=(5, 5), pady=(70, 10))#padx=150, pady=160, anchor='w')
    
def remove_api(self):
    self.api_key = self.api_entry.get()
    encrypt_key(self.api_key, path=self.folder_api)
    self.api_label.grid_remove()
    self.api_entry.grid_remove()
    self.button_ok.grid_remove()
    self.create_button_api()
    self.regular_app_layout()
    
def active_mode(self):
    # Set background image
    filepath = os.path.join(self.folder_images, 'cover-rec.png')
    self.bg_image=CTkImage(Image.open(filepath), size=self.monitor_size(0.37, 0.35))
    self.bg_label = ctk.CTkLabel(self, image=self.bg_image, text='')
    self.bg_label.place(relx=0, rely=0)
    self.button_enter.grid_remove()
    self.drop_model.grid_remove()
    self.label_model.grid_remove()
    self.button_talk.grid_remove()
    self.button_preprompt.grid_remove()
    self.check_autoreload.grid_remove()

    if not hasattr(self, "img_record"):
        self.img_record = create_icon(self, "recording.png")
    self.button_talk = ctk.CTkButton(self, image=self.img_record.image, border_width=0 , 
                                 bg_color="#13273a", fg_color="#13273a",
                                 width=20, height=20,
                                 command=lambda:stop_active_mode(self))
    self.button_talk.grid(row = 8, column = 9, padx=(0, 0), pady=(60, 10))
    ToolTip(self.button_talk, msg="Close the active conversation mode.", delay=1.0)

    self.start_talk()

def stop_active_mode(self):
    self.my_cortana.flag = False
    # If spinner is still running, stop it
    if hasattr(self.my_cortana, 'spinner'):
        if self.my_cortana.spinner.running:           
            self.my_cortana.spinner.stop()  
    
    # Destroy the second window
    if self.toplevel_window is not None:
        self.toplevel_window.destroy()
        self.toplevel_window = None
    # Redirect output towards files
    self.markdown_output = MarkdownOutput(self.filename)
    sys.stdout = self.markdown_output   
    self.button_param.grid_remove()
   
    # Set background image
    filepath = os.path.join(self.folder_images, 'cover.jpg')
    self.bg_image = CTkImage(Image.open(filepath), size=self.monitor_size(0.37, 0.35))
    self.bg_label = ctk.CTkLabel(self, text='', image=self.bg_image)
    self.bg_label.place(relx=0, rely=0)
    self.create_prompt()
    regular_app_layout(self)
