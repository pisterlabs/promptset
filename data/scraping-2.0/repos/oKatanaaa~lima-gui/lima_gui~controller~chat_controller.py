from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtCore import QTimer
import time
from functools import partial
import json
from typing import Optional

from lima_gui.view.chat_window import ChatWindow
from lima_gui.view.chat_item import ChatItem
from lima_gui.view.function_desc_window import FunctionDescriptionWindow
from lima_gui.model.chat import Chat, Function
from lima_gui.state.settings import Settings
from lima_gui.state.openai import OpenAIService
from lima_gui.controller.function_desc_controller import FunctionDescController


class ChatItemUpdater(QThread):
    update_period_sec = 0.25
    update_signal = Signal(str, object, object)
    
    def __init__(self, chat_item: ChatItem, conversation, assistant_role, functions=[]):
        super().__init__()
        self.chat_item = chat_item
        self.openai_service = OpenAIService.get_instance()
        self.conversation = conversation
        self.assistant_role = assistant_role
        self.functions = functions if len(functions) > 0 else None
        self.update_signal.connect(self.update_text)
    
    def run(self):
        context = self.chat_item.get_cursor_context()
        before, after = context
        if self.openai_service.get_api_type() == OpenAIService.API_TYPE_CHAT:
            before, after = '', ''
            
        def text_postprocessor(text, function_name, arguments):
            content = before + text + after
            if function_name:
                content += '\n'
                content += '\n---function call (this text will be removed once generation is finished)---'
                content += '\nfunction_name: ' + function_name
                content += '\narguments:\n' + arguments
            return content
        
        text = ''
        function_name = None
        arguments = ''
        time_past = 0.0
        for delta_time, chunk in self.openai_service.generate_response(
            self.conversation, context, self.functions):
            print(chunk)
            if isinstance(chunk, str):
                text += chunk
            elif isinstance(chunk, dict):
                if 'name' in chunk:
                    function_name = chunk['name']
                if 'arguments' in chunk:
                    arguments += chunk['arguments']
            time_past += delta_time
            if time_past > self.update_period_sec:
                self.update_signal.emit(text_postprocessor(text, function_name, arguments), None, None)
                time_past = 0.0
        self.update_signal.emit(text_postprocessor(text, None, None), function_name, arguments)
        # if function_name:
        #     self.update_function_call(function_name, arguments)

    def update_text(self, text, function_name, arguments):
        function_data = None
        if function_name:
            print('arguments', arguments)
            function_data = {
                'name': function_name,
                'arguments': json.loads(arguments)
            }
        self.chat_item.set_data(self.assistant_role, text, function_call_data=function_data, no_callback=False)
        
    def update_function_call(self, function_name, arguments):
        print('setting function call data')
        function_data = {
            'name': function_name,
            'arguments': json.loads(arguments)
        }
        self.chat_item.set_function_call_data(function_data)
        

class ChatController:
    def __init__(self, chat_window: ChatWindow, chat: Chat):
        print('created chat controller')
        self.chat_window = chat_window
        self.chat = chat
        
        self.settings = Settings.get_instance()
        self.openai_service = OpenAIService.get_instance()
        
        self.chat_window.set_name(chat.name)
        self.chat_window.set_language_options(self.settings.languages)
        self.chat_window.set_language(chat.language)
        self.chat_window.set_role_options(['system', 'user', 'assistant', 'function'], 'assistant')
        self.chat_window.set_functions(chat.functions)
        for msg in chat.chat['dialog']:
            role, content, fn_call_data = msg['role'], msg['content'], msg.get('function_call')
            self.chat_window.add_msg(role, content, fn_call_data)

            
        self.chat_window.set_token_count(self.settings.get_token_count(self.chat.to_str()))
        self.chat_window.set_tag_options(self.settings.tags)
        self.chat_window.set_tags(chat.tags)
        self.chat_window.set_msg_count(len(chat.chat['dialog']))
        self.chat_window.is_generate_allowed = self.openai_service.enabled
        print('created chat controller, openai service enabled: ', self.openai_service.enabled)

        self.chat_window.set_add_msg_clicked_callback(self.on_add_msg_clicked)
        self.chat_window.set_delete_msg_clicked_callback(self.on_delete_msg_clicked)
        self.chat_window.set_msg_changed_callback(self.on_msg_changed)
        self.chat_window.set_name_changed_callback(self.on_name_changed)
        self.chat_window.set_language_changed_callback(self.on_language_changed)
        self.chat_window.set_tag_added_callback(self.on_tag_added)
        self.chat_window.set_tag_deleted_callback(self.on_tag_deleted)
        self.chat_window.set_generate_callback(self.on_generate_clicked)
        self.chat_window.set_add_function_callback(self.on_function_add_clicked)
        self.chat_window.set_delete_function_callback(self.on_function_delete_clicked)
        self.chat_window.set_function_double_clicked_callback(self.on_function_double_clicked)
        
        # Make sure the list is scrolled to the bottom.
        # There is a problem that the chat items are not rendered at the time of adding
        # which makes their sizes wrong (they get correct sizes after rendering) and it breaks scrolling.
        QTimer.singleShot(100, self.chat_window.ui.listWidget.scrollToBottom)
        
        self.chat_item_updater = None
    
    def on_add_msg_clicked(self):
        print('add msg clicked')
        last_role = self.chat.last_role
        last_msg = self.chat.last_msg
        if last_role == 'system':
            next_role = 'user'
        elif last_role == 'user':
            next_role = 'assistant'
        elif last_role == 'assistant':
            if 'function_call' in last_msg:
                next_role = 'function'
            else:
                next_role = 'user'
        elif last_role == 'function':
            next_role = 'assistant'
        else:
            next_role = 'system'
        
        self.chat.add_msg(next_role, '')
        
        self.chat_window.add_msg(next_role, '')
        self.chat_window.set_msg_count(len(self.chat.chat['dialog']))
        
    def on_msg_changed(self, ind, role, content, fn_call_data):
        print('msg changed')
        print(fn_call_data)
        
        # Do not add fn_call_data if it is empty
        if fn_call_data['name'] is None:
            fn_call_data = None

        self.chat.edit_msg(ind, role, content, fn_call_data)
        self.chat_window.set_token_count(self.settings.get_token_count(self.chat.to_str()))
    
    def on_delete_msg_clicked(self, ind):
        print('delete msg clicked')
        self.chat.remove_msg(ind)
        self.chat_window.set_msg_count(len(self.chat.chat['dialog']))

    def on_name_changed(self, name):
        print('name changed')
        self.chat.name = name
    
    def on_language_changed(self, lang):
        print('language changed')
        self.chat.language = lang
        
    def on_tag_added(self, tag):
        print('tag added')
        self.chat.add_tag(tag)
        
    def on_tag_deleted(self, tag):
        print('tag deleted')
        self.chat.remove_tag(tag)
        
    def on_generate_clicked(self, ind, chat_item):
        print('generate clicked')
        conversation = self.chat.get_conversation_history(ind)
        self.chat_item_updater = ChatItemUpdater(
            chat_item=chat_item,
            conversation=conversation,
            assistant_role='assistant',
            functions=self.chat.functions
        )
        self.chat_item_updater.start()
        
    def on_function_add_clicked(self):
        function = Function.create_empty('new_function')
        fn_window = FunctionDescriptionWindow()
        fn_controller = FunctionDescController(fn_window, function)
        fn_controller.save_function_callback = self.on_function_created
        
        self.fn_window = fn_window
        self.fn_controller = fn_controller
        fn_window.show()
    
    def on_function_created(self, function: Function):
        print('function created')
        print(function)
        self.chat.add_fn(function)
        self.chat_window.set_functions(self.chat.functions)
        self.fn_window.close()
        self.fn_controller = None
        self.fn_window = None
    
    def on_function_delete_clicked(self, ind):
        self.chat.remove_fn(ind)
        self.chat_window.set_functions(self.chat.functions)
        
    def on_function_double_clicked(self, ind):
        function = self.chat.get_fn(ind)
        print('fn double clicked, name', function.name)
        
        fn_window = FunctionDescriptionWindow()
        fn_controller = FunctionDescController(fn_window, function)
        fn_controller.save_function_callback = partial(self.on_function_updated, ind)
        
        self.fn_window = fn_window
        self.fn_controller = fn_controller
        fn_window.show()

    def on_function_updated(self, ind, function):
        print('fn updated, name', function.name)
        print(function)
        self.chat.edit_fn(ind, function)
        self.fn_window.close()
        self.fn_controller = None
        self.fn_window = None
        
        self.chat_window.set_functions(self.chat.functions)
