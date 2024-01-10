import tkinter as tk
from tkinter import ttk, simpledialog
from markdown import markdown
from tkhtmlview import HTMLLabel
import asyncio
import aiofiles
import os
import subprocess

import tkinter.filedialog

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import mimetypes
import openai

from ttkthemes import ThemedTk

import time
from datetime import datetime, timezone

chat_model = 'gpt-4'

def chat_with_gpt(model, messages):
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response['choices'][0]['message']['content'].strip()
        
def generate_tell(ask_responses):
    tell_string = "+TELL\n"
    for var, data in ask_responses.items():
        ask_type = data['type']
        value = data['value']
        if ask_type == 'Ask':
            tell_string += f"Tell\t{var}={value}\r\n"
        elif ask_type in ['Choose', 'Select']:
            choices = value
            # If the value is a list, convert to comma separated
            if isinstance(choices, list):
                choices = ','.join(choices)
            tell_string += f"Choices\t{var}={choices}\r\n"
    return tell_string + ".\r\n"

def create_multipart_message(ask_responses):
    msg = MIMEMultipart()
    for var, data in ask_responses.items():
        ask_type = data['type']
        value = data['value']
        if ask_type in ["Ask", "Choose"]:
            msg.attach(MIMEText(f'Content-Disposition: form-data; name="{var}"\r\n\r\n{value}'))
        elif ask_type == "ChooseFile":
            if os.path.isfile(value):
                mime_type, _ = mimetypes.guess_type(value)
                if mime_type is None:
                    mime_type = "application/octet-stream"
                main_type, sub_type = mime_type.split("/", 1)
                with open(value, "rb") as file:
                    if main_type == "text":
                        file_data = MIMEText(file.read().decode(), _subtype=sub_type)
                    else:
                        file_data = MIMEBase(main_type, sub_type)
                        file_data.set_payload(file.read())
                        encoders.encode_base64(file_data)
                file_data.add_header("Content-Disposition", "form-data", name=var, filename=os.path.basename(value))
                msg.attach(file_data)
            else:
                msg.attach(MIMEText(f'Content-Disposition: form-data; name="{var}"\r\n\r\n{value}'))
        elif ask_type == "Select":
            for choice in value:
                msg.attach(MIMEText(f'Content-Disposition: form-data; name="{var}"\r\n\r\n{choice}'))
    return msg.as_string()

# Need to adjust so it can connect to different servers and also can connect through SSL.
class GopherClient:

    ITEM_TYPES = {
        "0": "(TEXT)",
        "1": "(DIR)",
        "9": "(BIN)",
        "I": "(IMAGE)",
        "<": "(SOUND)",
        ";": "(VIDEO)",
        "d": "(DOC)",
        "h": "(HTML)",
        "7": "(SEARCH)",
        "?": "(INTERACTIVE)", # indicates to gopher+ that there will be a +ASK block
    }

    DEFAULT_PORT = 70

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.menu = []
        self.location = ''
        self.menu_history = []  # stack to store menu history
        self.last_query = ['', None]  # to store last query for re-fetching pending interaction

    async def set_location(self, location, auto_fetch = True, host = None, port = None):
        self.location = location
        if host is not None:
            self.host = host
        if port is not None:
            self.port = self.DEFAULT_PORT
        if auto_fetch:
            self.fetch(self.location)

    async def fetch(self, selector, query = None, download = False, save = True, wait = False):
        print(f"Fetching {selector}...")
        self.last_query = [selector, query]
        reader, writer = await asyncio.open_connection(self.host, self.port)
        if query is not None:
            message = self.location + selector + '\t' + query + "\r\n"
        else:
            message = self.location + selector + "\r\n"
        writer.write(message.encode('utf-8'))
        await writer.drain()

        if wait:
            # The first line should be a +ASK
            line = await reader.readline();
            print(line)
            if line != b'+ASK\n':
                print("INVALID")
                return
            ask_inputs = []
            while True:
                # Doesn't check for invalid options, or correct formatting yet.
                ask = (await reader.readline()).decode('utf-8').rstrip()
                if not ask or ask == '.':
                    break

                ask_type, rest = ask.split("\t", 1)
                ask_var, ask_prompt = rest.split("=", 1)
                ask_choices = None

                if "\t" in ask_prompt:
                    ask_prompt, ask_choices = ask_prompt.split("\t", 1)

                if ask_choices is not None:
                    ask_choices = ask_choices.split(',')
                input = (ask_type, ask_prompt, ask_var, ask_choices)
                ask_inputs.append(input)

            dialog = AskDialog(self.root, "Input", ask_inputs)
            ask_responses = dialog.results
            # If the +ASK block contains a file, which therefore could be a large binary file, we'll use multipart. Otherwise we'll use tab separated as is closer to the original Gopher protocol.
            multipart = False
            for input in ask_inputs:
                if input[0] == 'ChooseFile':
                    multipart = True

            message = ""
            if multipart:
                message = create_multipart_message(ask_responses)
            else:
                message = generate_tell(ask_responses)

            print(message)
            writer.write(message.encode('utf-8'))
            await writer.drain()

        temp_menu = []  # temporary menu to store fetched items
        
        data = None
        if download:
            print(type)
            filename = selector.split('/')[-1:][0]

            # I don't think this is set up correctly. Might only work with text files. Yeah this definitely does not work with binary files. 
            if save:
                async with aiofiles.open(filename, 'wb') as fd:
                    while True:
                        line = await reader.readline()
                        if not line or line == b'.':
                            break
                        data += line + "\n"
                        await fd.write(line)

                if (os.name == 'nt'):  # For Windows
                    os.startfile(filename)
                elif (os.name == 'posix'):  # For Unix or Linux
                    subprocess.call(('xdg-open', filename))
            else:
                while True:
                    line = await reader.readline()
                    if not line or line == b'.':
                        break
                    data += line + "\n"

        else:
            if not query and selector != '':
                self.location += selector + '/'

            item_info = None
            data = ""
            while True:
                line = await reader.readline()
                if not line or line == b'.':
                    break
                line = line.decode('utf-8').rstrip()
                data += line + "\n"
                if line.startswith('+INFO: '):
                    item_info = line[7:].split('\t')
                else:
                    parts = line.split('\t')
                    item = [parts[0][0], parts[0][1:]] + parts[1:]
                    if item_info is not None:
                        item.extend(item_info)
                    item_info = None
                    temp_menu.append(item)

        writer.close()
        if temp_menu:  # if the fetched menu is not empty
            self.menu = temp_menu  # set current menu to the fetched menu
            self.menu_history.append((self.location, self.menu))  # add fetched menu to the history
            if len(self.menu_history) > 1:
                self.back_button.config(state=tk.NORMAL)  # enable back button
        await writer.wait_closed()

        return data

    def start(self):
#        asyncio.run(gc.fetch(''))
        root = ThemedTk(theme="yaru")
        root.title("Gopher+ Client")
        style = ttk.Style()
        style.configure("Treeview", rowheight=30) # increase row height
        tree = ttk.Treeview(root)

        self.root = root
        self.tree = tree

        # Create a scrollbar and attach it to tree
        scrollbar = tk.Scrollbar(root)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=tree.yview)

        tree["columns"]=("Filetype", "Filename", "Size", "Timestamp", "Description")
        tree.column("#0", width=0, stretch=tk.NO)
        tree.heading("#0", text='', anchor=tk.W)

        for col in tree["columns"]:
            tree.column(col, anchor=tk.W)
            tree.heading(col, text=col, anchor=tk.W)

        # Add items to treeview
        self.populate_tree()

        tree.pack(side=tk.TOP,fill=tk.X)

        back_button = tk.Button(root, text="Back", command=self.go_back)
        back_button.pack()
        self.back_button = back_button  # keep a reference to the back_button

        chat_button = tk.Button(root, text="Chat", command=self.chat_window)
        chat_button.pack()

        if len(self.menu_history) <= 1:
            self.back_button.config(state=tk.DISABLED)  # Initially disable the back button

        def on_select(event):
            item = tree.selection()[0]
            values = tree.item(item)['values']

            dt = values[0]
            filename = values[1]

            if dt == '(DIR)':
                asyncio.run(self.fetch(filename))
            elif dt == '(SEARCH)':
                query = simpledialog.askstring("Search", "Enter your search:")
                if query is not None:
                    asyncio.run(self.fetch(filename, query))
            elif dt == "(INTERACTIVE)":
                # Get the ASK inputs then render a dialogue, get input, and respond, then it'll generate the response.
                asyncio.run(self.fetch(filename, download = True, wait = True))
            else:
                asyncio.run(self.fetch(filename, download = True))

            root.after(100, self.populate_tree)

        tree.bind('<Double-1>', on_select)  # Bind double click to on_select
        root.mainloop()

    def populate_tree(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for item in self.menu:
            filetype = self.ITEM_TYPES.get(item[0], "(UNKNOWN)")
            filename = item[1]
            description = item[2]
            size = item[9] if item[9] != "-1" else ""
            timestamp = item[10] if item[10] else ""
            self.tree.insert('', 'end', values=(filetype, filename, size, timestamp, description))

    def go_back(self):
        if len(self.menu_history) > 1:  # ensure that we have a history to go back to
            self.menu_history.pop()  # remove current menu from history
            self.location = self.menu_history[-1][0]
            self.menu = self.menu_history[-1][1]  # set menu to last menu in history
            self.populate_tree()  # populate treeview
            if len(self.menu_history) <= 1:
                self.back_button.config(state=tk.DISABLED)  # disable the back button if we are at the start

    def link_click(self, event):
        clicked_item = self.chat_box.tk.call(self.chat_box._w, "href", event.x, event.y)
        if clicked_item:
            print(f"Clicked on link: {clicked_item}")

    def chat_window(self):
        if not hasattr(self, 'chat_history'):
            self.chat_history = []  # store chat history as list of tuples (source, text)

        chat_root = tk.Toplevel(self.root)  # create a new window 
        chat_root.title('Gopher Chat')

        # Create HTMLLabel widget to display chat history
        self.chat_box = HTMLLabel(chat_root, html="")
        self.chat_box.tag_bind("a", "<Button-1>", self.link_click)
        self.chat_box.pack(fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(chat_root)
        input_frame.pack(fill=tk.X)

        self.input_box = tk.Entry(input_frame)
        self.input_box.grid(row=0, column=0, sticky=tk.W+tk.E)
        self.input_box.bind('<Return>', lambda _: self.process_prompt())  # Bind Return key

        chat_button = tk.Button(input_frame, text='Send', command=self.process_prompt)
        chat_button.grid(row=0, column=1)

        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=0)

        self.update_chat_box()
        chat_root.mainloop()

    
    def process_prompt(self):
        prompt = self.input_box.get()
        if prompt:
            self.chat_history.append(('User', prompt))  # add user prompt to history
            self.input_box.delete(0, tk.END)  # clear input box

            # Check if we need to gather additional information. 
            while True:
                # I need the instructions very specific, but maybe I can break it up into multiple system prompts, and maybe tack on temporary system prompts saying when the system misbehaves.
                system_message = {"role": "system", "content": f"You are a helpful assistant. This is the data collection phase. Before you can repsond to the user, you must double check to make sure you don't need to perform any internal commands. You can perform a number of operations, or just answer questions in general. The following is some general information.\n\nServer: {self.host}:{self.port}\n\nDate and Time (UTC): {datetime.now(timezone.utc)}\n\nCurrent Directory: {self.location}\n\nDo you need to perform any additional functions before answering responding to the user? Here are the following options:\n- fetch [path, without leading /, so just fetch if you're fetching root]: fetches an item from the current gopher server.\n- hop [host] [port] - hop to a different Gopher server\n- none (by itself): specifies that there's no need for additional information and to continue to the response stage.\n Please select a command to execute or none. Your response must start with a valid command. You will have a chance to write a full response after your data collection has been completed by selecting **none**. "}
                messages = [{"role": role.lower(), "content": content} for role, content in self.chat_history]
                messages.insert(0, system_message)            

                # Process prompt.
                response = chat_with_gpt(chat_model, messages)
                # Split the response by space
                parts = response.split(' ', 1)  # The second argument (1) ensures the string is split at the first occurrence of a space

                # Set command and parameters
                command = parts[0].lower()
                print(command)
                parameters = ""

                # Check if there are any parameters (i.e., the string contained at least one space)
                if len(parts) > 1:
                    parameters = parts[1]
                    
                if command == "none":
                    break
                else:
                    result = None
                    # I need to add a way to have it choose whether to download or not.
                    if command == "fetch":
                        # The system doesn't really know what kind of data it's getting for a fetch, so I should tack on an explanation or something, or convert to an easier to understand format.
                        asyncio.run(gc.fetch(parameters))
                        # Tell the AI what each entry is. 
                        result = "Item Type | Filename | Selector | Host | Port | Item Type (again) | Short Description | Description | Mime Type | Size | Last Modified\n"
                        for item in self.menu:
                            print(item)
                            result += ' | '.join(item) + '\n'
                    if result is not None:
                        self.chat_history.append(('System', f"System command executed... {response}\nResult:\n{result}"))
                    else:
                        # Unknown command must have been interested. Ignored. 
                        pass
            # Would really be great to bring back my personality profile system from SAM. 
            system_message = {"role": "system", "content": f"You are a helpful assistant. The following is some general information.\n\nServer: {self.host}:{self.port}\n\nDate and Time (UTC): {datetime.now(timezone.utc)}\n\nCurrent Directory: {self.location}\n\nRespond to the most recent user prompt. Be polite, and semi formal, but not obnoxious or stuck up. Use full markdown."}
            messages = [{"role": role.lower(), "content": content} for role, content in self.chat_history]
            messages.insert(0, system_message)            
            response = chat_with_gpt(chat_model, messages)
            self.chat_history.append(('Assistant', response))  # add assistant prompt to history

            # Update chat box with the new history
            self.update_chat_box()

    def update_chat_box(self):
        new_html = "<!DOCTYPE html><html><body>"
        for role, message in self.chat_history:
            if role != "System":
                # Convert Markdown message to HTML
                message_html = markdown(message)
                # Add message to new HTML
                new_html += f"<p><b>{role}</b>: {message_html}</p>"
        new_html += "</body></html>"
        # Update the HTMLLabel with the new HTML
        self.chat_box.set_html(new_html)

# Need to adjust to have the variable type and then actually grab the file and its MIME type.
class AskDialog(tk.simpledialog.Dialog):
    def __init__(self, parent, title=None, ask_inputs=None):
        self.ask_inputs = ask_inputs
        self.results = []
        tk.simpledialog.Dialog.__init__(self, parent, title)

    def body(self, master):
        self.widgets = []
        for i, (ask_type, ask_prompt, ask_var, ask_choices) in enumerate(self.ask_inputs):
            tk.Label(master, text=ask_prompt).grid(row=i)
            if ask_type == "Ask":
                entry = tk.Entry(master)
                entry.grid(row=i, column=1)
                self.widgets.append(entry)
            elif ask_type == "Choose":
                if ask_choices:
                    ask_choices_list = ask_choices
                    var = tk.StringVar(value=ask_choices_list[0])
                    dropdown = tk.OptionMenu(master, var, *ask_choices_list)
                    dropdown.grid(row=i, column=1)
                    self.widgets.append((dropdown, var))
            elif ask_type == "Select":
                if ask_choices:
                    ask_choices_list = ask_choices
                    listbox = tk.Listbox(master, selectmode=tk.MULTIPLE)
                    listbox.grid(row=i, column=1)
                    for choice in ask_choices_list:
                        listbox.insert(tk.END, choice)
                    self.widgets.append(listbox)
            elif ask_type == "ChooseFile":
                var = tk.StringVar()
                button = tk.Button(master, text="Select File", command=lambda: var.set(tk.filedialog.askopenfilename()))
                button.grid(row=i, column=1)
                self.widgets.append((button, var))
        return self.widgets[0]

    def apply(self):
        results = {}
        for i, widget in enumerate(self.widgets):
            ask_type, _, ask_var, _ = self.ask_inputs[i]  # get ask_type and ask_var
            result = {}
            if isinstance(widget, tk.Entry):
                result["type"] = "Ask"
                result["value"] = widget.get()
            elif isinstance(widget, tuple) and isinstance(widget[0], tk.Button):
                result["type"] = "ChooseFile"
                result["value"] = widget[1].get()
            elif isinstance(widget, tk.Listbox):
                result["type"] = "Select"
                result["value"] = [widget.get(i) for i in widget.curselection()]
            else:  # OptionMenu for "Choose"
                result["type"] = "Choose"
                result["value"] = widget[1].get()
            results[ask_var] = result
        self.results = results
        
gc = GopherClient('localhost', 10070)
gc.start()
