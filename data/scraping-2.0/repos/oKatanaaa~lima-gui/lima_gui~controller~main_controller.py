import pandas as pd
from functools import partial
from copy import deepcopy

from lima_gui.view.main_window import MainWindow
from lima_gui.view.chat_window import ChatWindow
from lima_gui.view.settings_window import SettingsWindow
from lima_gui.model.chat_dataset import ChatDataset
from lima_gui.model.chat import Chat
from .chat_controller import ChatController
from .settings_controller import SettingsController
from .openai_controller import OpenAIController


class Controller:
    def __init__(self, main_window: MainWindow):
        self.main_window = main_window
        self.openai_controller = OpenAIController(self.main_window.openai_window)
        self.main_window.set_column_names(['Chat name', 'Language', 'Msg count'])
        
        self.main_window.set_add_chat_callback(self.on_add_chat_clicked)
        self.main_window.set_delete_chat_callback(self.on_delete_chat_clicked)
        self.main_window.set_chat_double_clicked_callback(self.on_chat_double_clicked)
        self.main_window.set_save_callback(self.on_save_triggered)
        self.main_window.set_open_callback(self.on_open_triggered)
        self.main_window.set_settings_callback(self.on_settings_clicked)
        self.main_window.set_copy_chat_callback(self.on_copy_chat_clicked)
        self.main_window.show()
        
        self.dataset = ChatDataset([])
        
        self.chat_controllers = set()
    
    def on_save_triggered(self, filename):
        print('save triggered', filename)
        pd = self.dataset.to_pandas()
        pd.to_csv(filename, index=False)
        
    def on_open_triggered(self, filename):
        print('open triggered', filename)
        self.dataset = ChatDataset.from_pandas(pd.read_csv(filename))
        self.update_table()
        
    def on_add_chat_clicked(self):
        print('add chat clicked')
        chat = Chat.create_empty()
        self.dataset.add_chat(chat)
        self.main_window.add_chat_item([
            chat.name, chat.language, len(chat)])
        
    def on_delete_chat_clicked(self, row_id):
        print('delete chat clicked', row_id)
        self.dataset.remove_chat(row_id)
    
    def on_copy_chat_clicked(self, row_id):
        print('copy chat clicked', row_id)
        if row_id == -1:
            return
        
        chat = self.dataset.get_chat(row_id)
        chat = deepcopy(chat)
        chat.name += ' (copy)'
        self.dataset.add_chat(chat)
        self.main_window.add_chat_item([
            chat.name, chat.language, len(chat)])
        
    def on_chat_double_clicked(self, row_id):
        print('chat double clicked', row_id)
        chat_window = ChatWindow()
        chat_controller = ChatController(
            chat_window, self.dataset.get_chat(row_id))
        self.chat_controllers.add(chat_controller)
        
        chat_window.set_close_callback(
            partial(self.on_chat_window_close, chat_controller=chat_controller))
        chat_window.show()
    
    def on_chat_window_close(self, chat_controller):
        print('chat window closed')
        self.chat_controllers.remove(chat_controller)
        self.update_table()
        
    def on_settings_clicked(self):
        settings_window = SettingsWindow()
        self.settings_controller = SettingsController(settings_window)
        
        settings_window.set_close_callback(self.on_settings_window_close)
        settings_window.show()
        
    def on_settings_window_close(self):
        print('settings window closed')
        self.settings_controller = None
        self.update_table()
        
    def update_table(self):
        self.main_window.clear()
        for i in range(len(self.dataset)):
            chat = self.dataset.get_chat(i)
            self.main_window.add_chat_item([
                chat.name, chat.language, len(chat)])
