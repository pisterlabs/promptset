import wx
import openai
import threading
import time
import json
from datetime import datetime
from collections import deque
import os
import subprocess
from key import key
import atexit
import sys
import shutil

# sys.stderr = open('./log.txt', 'w')

VERSION = "2.0"

# Initialize the OpenAI API
# sb_base = "https://api.openai-sb.com/v1"

openai.api_key = key
# openai.api_base = sb_base

# Define a new event type for UI updates
wxEVT_UPDATE_UI = wx.NewEventType()
EVT_UPDATE_UI = wx.PyEventBinder(wxEVT_UPDATE_UI, 1)

def find_python():
    # Attempt to find Python in PATH
    python_path = shutil.which("python3") or shutil.which("python")
    if python_path:
        return python_path

    # Check common paths
    common_paths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python",
        "/usr/local/bin/python",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path

    # Return None if Python wasn't found
    return None

python_path = find_python()

class ChatApp(wx.App):
    def OnInit(self):
        frame = ChatFrame(None, "Chirp")
        frame.Show()
        return True
    
class UpdateUIEvent(wx.PyCommandEvent):
    def __init__(self, etype, eid, value=None):
        super(UpdateUIEvent, self).__init__(etype, eid)
        self._value = value

    def GetValue(self):
        return self._value
    
class ChatSession:
    def __init__(self, chat_id, messages=[]):
        self.id = chat_id
        self.date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.messages = messages

    def to_dict(self):
        return {
            "id": self.id,
            "date": self.date,
            "messages": self.messages
        }

    @classmethod
    def from_dict(cls, data):
        chat_session = cls(data["id"], data["messages"])
        chat_session.date = data["date"]
        return chat_session

class ChatFrame(wx.Frame):
    def __init__(self, parent, title):
        super(ChatFrame, self).__init__(parent, title=title, size=(700, 700))

        atexit.register(self.terminate_local_model_server)

        panel = wx.Panel(self)
        sizer_main = wx.BoxSizer(wx.HORIZONTAL)
        # sizer_hist = wx.BoxSizer(wx.VERTICAL)
        sizer_chat = wx.BoxSizer(wx.VERTICAL) # Enclose the current chat UI in a vertical sizer
        
        # Add the chat history log as a list control on the left.
        self.chat_history_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.chat_history_list.InsertColumn(0, 'Date', width=150)

        # Add system prompt label and control
        self.system_prompt_label = wx.StaticText(panel, label="System Prompt:")
        default_prompt = "You are a helpful assistant."
        self.system_prompt_ctrl = wx.TextCtrl(panel, value=default_prompt, style=wx.TE_MULTILINE)

        # Adjust the sizers to include the system prompt control
        sizer_system_prompt = wx.BoxSizer(wx.VERTICAL)
        sizer_system_prompt.Add(self.system_prompt_label, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        sizer_system_prompt.Add(self.system_prompt_ctrl, proportion=0, flag=wx.EXPAND | wx.ALL, border=5) # change proportion to 0 to make it smaller

        # Clean history button
        self.clean_history_button = wx.Button(panel, label="Clean History")
        self.clean_history_button.Bind(wx.EVT_BUTTON, self.on_clean_history)

        # Open history button
        self.open_history_button = wx.Button(panel, label="Open History Folder")
        self.open_history_button.Bind(wx.EVT_BUTTON, self.on_open_history)


        sizer_left = wx.BoxSizer(wx.VERTICAL)  # Sizer for the left side which contains chat history and the button
        sizer_left.Add(self.chat_history_list, 1, wx.EXPAND | wx.ALL, 5)
        sizer_left.Add(sizer_system_prompt, 0, wx.EXPAND | wx.ALL, 5)  # change proportion to 0 to make it smaller
        sizer_left.Add(self.clean_history_button, 0, wx.EXPAND | wx.ALL, 5)
        sizer_left.Add(self.open_history_button, 0, wx.EXPAND | wx.ALL, 5)


        sizer_main = wx.BoxSizer(wx.HORIZONTAL)
        sizer_chat = wx.BoxSizer(wx.VERTICAL)  # Enclose the current chat UI in a vertical sizer
        sizer_main.Add(sizer_left, 0, wx.EXPAND | wx.ALL, 5)
        sizer_main.Add(sizer_chat, 1, wx.EXPAND)
        panel.SetSizer(sizer_main)



        # RadioBox for model selection
        self.models = ['GPT', 'Local Model (GGUF)']
        self.model_selector = wx.RadioBox(panel, label="Choose Model", choices=self.models, majorDimension=1, style=wx.RA_SPECIFY_ROWS)
        self.model_selector.Bind(wx.EVT_RADIOBOX, self.on_model_selection)
        sizer_chat.Add(self.model_selector, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        self.model = 'gpt-3.5-turbo'

        self.conversation_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.input_ctrl = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER, size=(400, 100))
        self.send_button = wx.Button(panel, label="Send")
        self.paste_button = wx.Button(panel, label="Paste")

        sizer_chat.Add(self.conversation_ctrl, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)
        sizer_chat.Add(self.input_ctrl, proportion=0, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)
        sizer_chat.Add(self.send_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)
        sizer_chat.Add(self.paste_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        self.send_button.Bind(wx.EVT_BUTTON, self.on_send)
        self.input_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_send)
        self.paste_button.Bind(wx.EVT_BUTTON, self.onPaste)
        self.input_ctrl.Bind(wx.EVT_TEXT_PASTE, self.onPaste)
        self.export_button = wx.Button(panel, label="Export History")
        self.export_button.Bind(wx.EVT_BUTTON, self.onExportHistory)
        sizer_chat.Add(self.export_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        self.new_chat_button = wx.Button(panel, label="Start New Chat")
        self.new_chat_button.Bind(wx.EVT_BUTTON, self.on_new_chat)
        sizer_chat.Add(self.new_chat_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        # Button to load model
        # self.load_model_button = wx.Button(panel, label="Load Model")
        # self.load_model_button.Bind(wx.EVT_BUTTON, self.on_load_model)
        # sizer_chat.Add(self.load_model_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)


        # helpMenu = wx.Menu()
        # self.versionMenuItem = helpMenu.Append(wx.ID_ANY, "Show Version", "Show application version")
        # self.Bind(wx.EVT_MENU, self.onShowVersion, self.versionMenuItem)

        # menubar = wx.MenuBar()
        # menubar.Append(helpMenu, '&Help')
        # self.SetMenuBar(menubar)
        # sizer.Add(menubar, 0, wx.EXPAND, parent=panel)


        # Initiate the conversation with a system message
        self.messages = [{"role": "system", "content": self.system_prompt_ctrl.GetValue()}]

        # Bind the update UI event to a handler
        self.Bind(EVT_UPDATE_UI, self.on_update_ui)

        self.SetTransparent(200)

        font = wx.Font(16, wx.DEFAULT, wx.NORMAL, wx.NORMAL, faceName="Helvetica")
        self.conversation_ctrl.SetFont(font)

        # OPEN_CHAT_SHORTCUT = wx.AcceleratorEntry(wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord('C')) 
        # if wx.Platform == '__WXMAC__':
        #     OPEN_CHAT_SHORTCUT = wx.AcceleratorEntry(wx.ACCEL_CMD | wx.ACCEL_CTRL, ord('C'))
        # else: 
        #     OPEN_CHAT_SHORTCUT = wx.AcceleratorEntry(wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord('C'))
        # accel_tbl = wx.AcceleratorTable([OPEN_CHAT_SHORTCUT])

        # self.SetAcceleratorTable(accel_tbl)
        # self.Bind(wx.EVT_MENU, self.on_open_chat, id=OPEN_CHAT_SHORTCUT.GetCommand())

        self.bot_color = wx.Colour(0, 120, 215)
        self.user_color = wx.Colour(255, 193, 7)

        self.bot_last_position = None
        self.start = True

        self.last_chat_id = -1
        self.current_chat_id = 0

        # history
        self.history_file = "chat_history.json"
        try:
            with open(self.history_file, 'r') as f:
                self.all_chats_json = json.load(f)
                self.all_chats = self.convert(self.all_chats_json)
                try:
                    self.last_chat_id = self.all_chats[-1].id if self.all_chats else -1
                except:
                    self.last_chat_id = -1
                self.current_chat_id = self.last_chat_id + 1
                self.last_chat_id = self.current_chat_id
        except FileNotFoundError:
            self.all_chats = deque()

        self.all_chats.appendleft(ChatSession(self.current_chat_id, self.messages))

        self.chat_history_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.onViewChat)
        self.update_history_list()

    def on_model_selection(self, event):
        selected_model = self.models[self.model_selector.GetSelection()]
        self.terminate_local_model_server()

        if selected_model == 'GPT':
            # Logic to switch to GPT model
            openai.api_base = "https://api.openai.com/v1"
            self.model = 'gpt-3.5-turbo'

        elif selected_model == 'Local Model (GGUF)':
            try:
                with wx.FileDialog(self, "Choose a model file", wildcard="Model files (*.gguf)|*.gguf", 
                                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
                    if fileDialog.ShowModal() == wx.ID_CANCEL:
                        return     # User cancelled the action

                # Proceed with loading the file
                self.model_file_path = fileDialog.GetPath()
                model_name = os.path.basename(self.model_file_path)
                # Construct the command to start the server
                # cmd = [
                #     "python", "-m",
                #     "llama_cpp.server", "--model", self.model_file_path
                # ]

                # server_script_path = os.path.join(os.path.dirname(sys.executable), 'llama_cpp', 'server', '__main__.py')

                base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                server_script_path = os.path.join(base_path, 'llama_cpp', 'server', '__main__.py')
                cmd = [python_path, server_script_path, "--model", self.model_file_path] #, "--n_gpu_layers", "-1"]
                # print(cmd)

                # Start the server as a background process
                self.server_process = subprocess.Popen(cmd)

                # Wait for a short while and check if the process is still running
                # This is a basic way to check for immediate failures in process start
                self.server_process.poll()
                if self.server_process.returncode is not None:
                    raise Exception("Server process terminated immediately after start.")

                # Logic to switch to local model using self.model_file_path
                openai.api_base = "http://localhost:8000/v1"
                self.model = self.model_file_path #f"../models/{model_name}"
            except Exception as e:
                # pop up a dialog to show error
                wx.MessageBox(f"Error: {str(e)}", "Model Loading Error")

        

    def terminate_local_model_server(self):
        """Helper method to terminate the local model server if it's running."""
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            del self.server_process  # Remove the attribute after terminating the process

    def on_close(self, event):
        """Handler to gracefully shut down the server subprocess when the application closes."""
        self.terminate_local_model_server()
        event.Skip()

    def on_clean_history(self, event):
        # Clear the history file
        with open("chat_history.json", "w") as file:
            file.write("[]")
        # Refresh the chat_history_list to reflect the cleared history
        self.chat_history_list.DeleteAllItems()
        self.all_chats = deque()
        self.messages = [{"role": "system", "content": self.system_prompt_ctrl.GetValue()}]
        self.current_chat_id = 0
        self.last_chat_id = self.current_chat_id
        self.start = True
        self.conversation_ctrl.SetValue('')
        self.all_chats.appendleft(ChatSession(self.current_chat_id, self.messages))
        self.save_current_chat_to_history()
        # Clear the UI chat window
        self.update_history_list()

    def on_open_history(self, event):
        history_folder = os.path.dirname(os.path.abspath("chat_history.json"))  # Get the folder of the history file
        # Open the folder using the default file explorer.
        if os.name == 'nt':  # Windows
            subprocess.Popen(f'explorer "{history_folder}"')
        elif os.name == 'posix':  # macOS
            subprocess.Popen(['open', history_folder])
        else:  # Linux and other UNIX
            subprocess.Popen(['xdg-open', history_folder])

    def convert(self, chat_list_json):
        ret = deque()
        for chat in chat_list_json:
            ret.append(ChatSession.from_dict(chat))
        return ret
    
    def convert_json(self, chat_list):
        ret = []
        for chat in chat_list:
            ret.append(chat.to_dict())
        return ret

    def update_history_list(self):
        # # Clear the existing listbox items
        self.chat_history_list.DeleteAllItems()
        # Add all chat IDs to the listbox
        for chat in self.all_chats:
            self.chat_history_list.Append([chat.date])
        self.Show()

    def on_new_chat(self, event):
        # Increase the chat ID
        self.current_chat_id = self.last_chat_id + 1
        self.last_chat_id = self.current_chat_id
        self.messages = [{"role": "system", "content": self.system_prompt_ctrl.GetValue()}]
        self.conversation_ctrl.SetValue('')
        self.start = True
        self.all_chats.appendleft(ChatSession(self.current_chat_id, self.messages))
        self.save_current_chat_to_history()
        # Clear the UI chat window
        self.update_history_list()


    def onViewChat(self, event):
        selected_index = event.GetIndex()
        selected_chat = self.all_chats[selected_index]
        self.messages = selected_chat.messages
        self.current_chat_id = selected_chat.id
        # Render the selected chat in the main conversation area
        self.conversation_ctrl.SetValue('')  # Clear current display
        self.showMessage()
    
    def showMessage(self):
        for msg in self.messages:
            role, content = msg['role'], msg['content']
            if role not in ['assistant', 'user']:
                continue
            color = self.bot_color if role == "assistant" else self.user_color
            self.conversation_ctrl.SetDefaultStyle(wx.TextAttr(color))
            self.conversation_ctrl.AppendText(f"{role.capitalize()}: {content}\n")

    def onExportHistory(self, event):
        dlg = wx.FileDialog(self, "Save chat history as...", defaultFile="chat_history.txt", 
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with open(path, 'w') as f:
                for msg in self.messages:
                    f.write(msg['role'] + ': ')
                    f.write(msg['content'] + '\n')
        dlg.Destroy()

    def onShowVersion(self, event):
        wx.MessageBox(f"Version: {VERSION}", "Version Info")

    def on_open_chat(self, event):
        self.Show()

    def on_send(self, event):
        self.messages[0] = {"role": "system", "content": self.system_prompt_ctrl.GetValue()}
        user_message = self.input_ctrl.GetValue()
        # Ensure there's a newline before the user's message
        if not self.conversation_ctrl.GetValue().endswith('\n'):
            if self.start:
                self.start = False
            else:
                self.conversation_ctrl.AppendText("\n")
        self.conversation_ctrl.SetDefaultStyle(wx.TextAttr(self.user_color))
        self.conversation_ctrl.AppendText(f"You: {user_message}\n")
        self.messages.append({"role": "user", "content": user_message})
        self.bot_last_position = None  # Reset for a new message

        threading.Thread(target=self.fetch_response).start()

    def onPaste(self, event):
        text = wx.TextDataObject()
        if wx.TheClipboard.Open():
            wx.TheClipboard.GetData(text)
            wx.TheClipboard.Close()
        self.input_ctrl.WriteText(text.GetText())
    
    def fetch_response(self):
        # print(self.messages)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=0.3,
            max_tokens=1024,
            stream=True
        )

        accumulated_response = []
        full_response = []
        last_update_time = time.time()

        for chunk in response:
            content = chunk['choices'][0]['delta'].get('content', '')
            accumulated_response.append(content)
            full_response.append(content)
            current_time = time.time()
            
            if current_time - last_update_time >= 0.1:  # Update UI every 100ms
                evt = UpdateUIEvent(wxEVT_UPDATE_UI, -1, ''.join(accumulated_response))
                wx.PostEvent(self, evt)
                accumulated_response.clear()  # Clear accumulated response after sending it to UI
                last_update_time = current_time

        # Send any remaining response chunks to UI
        if accumulated_response:
            evt = UpdateUIEvent(wxEVT_UPDATE_UI, -1, ''.join(accumulated_response))
            wx.PostEvent(self, evt)
        
        self.messages.append({"role": "assistant", "content": ''.join(full_response)})
        print(self.messages)
        self.save_current_chat_to_history()
        self.update_history_list()

    def save_current_chat_to_history(self):
        chat_session = self.all_chats[0]
        self.all_chats.popleft()
        chat_session.messages = self.messages
        self.all_chats.appendleft(chat_session)
        with open(self.history_file, 'w') as f:
            json.dump(self.convert_json(self.all_chats), f, indent=4)

    def on_update_ui(self, event):
        content = event.GetValue()

        if self.bot_last_position is None:
            # Ensure there's a newline before the bot's message
            if not self.conversation_ctrl.GetValue().endswith('\n'):
                self.conversation_ctrl.AppendText("\n")
            self.conversation_ctrl.SetDefaultStyle(wx.TextAttr(self.bot_color))
            self.conversation_ctrl.AppendText(f"Assistant: {content}")
            self.bot_last_position = self.conversation_ctrl.GetLastPosition()
        else:
            # Additional chunks from the bot for this message
            self.conversation_ctrl.SetInsertionPoint(self.bot_last_position)
            self.conversation_ctrl.SetDefaultStyle(wx.TextAttr(self.bot_color))
            self.conversation_ctrl.WriteText(content)
            self.bot_last_position = self.conversation_ctrl.GetLastPosition()

        self.input_ctrl.SetValue('')

if __name__ == "__main__":
    app = ChatApp(False)
    app.MainLoop()
