#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd() + "/gpt-stream-json-parser")

import openai
import json
import threading
from gpt_stream_parser import force_parse_json
import copy
import time

from src import extension

class MascotChatGpt:
    chatgpt_messages = []
    chatgpt_response = None
    log_file_name = None
    chatgpt_model_name = "gpt-3.5-turbo"
    chatgpt_functions = [{
        "name": "message_and_change_states",
        "description": """
Change the state of the character who will be speaking, then send the message.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "voice_style": {
                    "type": "string",
                    "description": "",
                },
                "eyebrow": {
                    "type": "string",
                    "description": "Change eyebrow (Either normal/troubled/angry/happy/serious).",
                },
                "eyes": {
                    "type": "string",
                    "description": "Change eyes (Either normal/closed/happy_closed/relaxed_closed/surprized/wink).",
                },
                "message": {
                    "type": "string",
                    "description": "Japanese message(lang:ja).",
                },
            },
            "required": ["message"],
        },
    }]
    recieved_message = ''
    recieved_states_data = None
    is_send_to_chatgpt = False
    last_time_chatgpt = 0.0

    def __init__(self, apikey):
        openai.api_key = apikey

    def load_model(self, model_name):
        self.chatgpt_model_name = model_name

    def load_log(self, log):
        if log is None:
            return False
        try:
            self.log_file_name = log
            if os.path.isfile(log):
                with open(log, 'r', encoding='UTF-8') as f:
                    self.chatgpt_messages = json.loads(f.read())
                return True
        except:
            pass
        return False

    def load_setting(self, chatgpt_setting, voicevox_style_names):
        self.chatgpt_messages = []
        if os.path.isfile(chatgpt_setting):
            with open(chatgpt_setting, 'r', encoding='UTF-8') as f:
                chatgpt_setting_content = f.read()
        else:
            chatgpt_setting_content = ''
        style_names_str = ''
        for style_name in voicevox_style_names:
            style_names_str += style_name
            if voicevox_style_names[-1] != style_name:
                style_names_str += '/'
        self.chatgpt_functions[0]["parameters"]["properties"]["voice_style"]["description"] = '''
Change voice style (Either ''' + style_names_str + ''').
        '''
        self.chatgpt_messages.append({"role": "system", "content": chatgpt_setting_content})

    def change_setting_from_str(self, chatgpt_setting_str):
        self.chatgpt_messages[0] = {"role": "system", "content": chatgpt_setting_str}

    def write_log(self):
        if self.log_file_name is None:
            return        
        with open(self.log_file_name + '.tmp', 'w', encoding='UTF-8') as f:
            f.write(json.dumps(self.chatgpt_messages, sort_keys=True, indent=4, ensure_ascii=False))
        if os.path.isfile(self.log_file_name):
            os.rename(self.log_file_name, self.log_file_name + '.prev')
        os.rename(self.log_file_name + '.tmp', self.log_file_name)
        if os.path.isfile(self.log_file_name + '.prev'):
            os.remove(self.log_file_name + '.prev')

    def send_to_chatgpt(self, content, write_log=True):
        #if self.chatgpt_response is not None:
        #    return False

        self.chatgpt_messages.append({"role": "user", "content": content})

        system_messages = self.chatgpt_messages[0]['content']
        all_funcs = self.chatgpt_functions
        for ext in extension.extensions:
            funcs = ext.get_chatgpt_functions()
            if funcs is not None:
                all_funcs = all_funcs + funcs
            mes = ext.get_chatgpt_system_message()
            if mes is not None:
                system_messages += '\n' + mes

        chatgpt_messages = copy.deepcopy(self.chatgpt_messages)
        chatgpt_messages[0]['content'] = system_messages

        def recv():
            self.recieved_message = ''
            recieved_json = ''
            self.recieved_states_data = None
            self.lock()
            self.chatgpt_response = openai.ChatCompletion.create(
                model=self.chatgpt_model_name,
                messages=chatgpt_messages,
                stream=True,
                functions=all_funcs
            )
            is_func = False
            func_name = None
            for chunk in self.chatgpt_response:
                if 'function_call' in chunk.choices[0].delta and chunk.choices[0].delta.function_call is not None:
                    if 'arguments' in chunk.choices[0].delta.function_call:
                        recieved_json += chunk.choices[0].delta.function_call.arguments
                        self.recieved_states_data = force_parse_json(recieved_json)
                    if 'name' in chunk.choices[0].delta.function_call:
                        func_name = chunk.choices[0].delta.function_call.name
                    if func_name == 'message_and_change_states':
                        if self.recieved_states_data is not None and 'message' in self.recieved_states_data:
                            self.recieved_message = self.recieved_states_data['message']
                        for ext in extension.extensions:
                            ext.recv_message_streaming(self.chatgpt_messages, self.recieved_message)
                    elif func_name is not None:
                        message = ''
                        for ext in extension.extensions:
                            message_part = ext.recv_function_streaming(self.chatgpt_messages, func_name, self.recieved_states_data)
                            if type(message_part) is str:
                                if message != '':
                                    message += '\n'
                                message += message_part
                            self.recieved_message = message
                        for ext in extension.extensions:
                            ext.recv_message_streaming(self.chatgpt_messages, self.recieved_message)
                        is_func = True
                else:
                    self.recieved_message += chunk.choices[0].delta.get('content', '')
                    for ext in extension.extensions:
                        ext.recv_message_streaming(self.chatgpt_messages, self.recieved_message)
            recieved_states_data = self.recieved_states_data
            self.unlock()
            resend_flag = False
            if is_func:
                message = ''
                for ext in extension.extensions:
                    resend_or_message = ext.recv_function(self.chatgpt_messages, func_name, recieved_states_data)
                    if type(resend_or_message) is str:
                        if message != '':
                            message += '\n'
                        message += resend_or_message
                    elif resend_or_message is not None:
                        resend_flag = True
                self.recieved_message = message
            if resend_flag:
                self.chatgpt_messages = self.chatgpt_messages[:-1]
                self.send_to_chatgpt(content, write_log)
            else:
                self.chatgpt_messages.append({"role": "assistant", "content": self.recieved_message})
                for ext in extension.extensions:
                    ext.recv_message(self.chatgpt_messages)
                if write_log:
                    self.write_log()
                self.chatgpt_response = None

        self.chatgpt_response = []
        recv_thread = threading.Thread(target=recv)
        recv_thread.start()

        return True

    def get_states(self):
        is_finished = self.chatgpt_response is None
        recieved_states_data = self.recieved_states_data
        voice_style = None
        eyebrow = None
        eyes = None
        if recieved_states_data is not None:
            if 'voice_style' in recieved_states_data:
                voice_style = recieved_states_data['voice_style']
            if 'eyebrow' in recieved_states_data:
                eyebrow = recieved_states_data['eyebrow']
            if 'eyes' in recieved_states_data:
                eyes = recieved_states_data['eyes']
        return is_finished, voice_style, eyebrow, eyes

    def get_message(self):
        return self.chatgpt_response is None, self.recieved_message

    def remove_last_conversation(self, result=None, write_log=True):
        if result is None or self.chatgpt_messages[-1]["content"] == result:
            self.chatgpt_messages = self.chatgpt_messages[:-2]
            if write_log:
                self.write_log()
            for ext in extension.extensions:
                ext.remove_last_conversation()

    def get_model_name(self):
        return self.chatgpt_model_name

    def lock(self):
        while self.is_send_to_chatgpt:
            time.sleep(0)
        self.is_send_to_chatgpt = True
        sleep_time = 0.5 - (time.time() - self.last_time_chatgpt)
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        self.last_time_chatgpt = time.time()

    def unlock(self):
        self.is_send_to_chatgpt = False