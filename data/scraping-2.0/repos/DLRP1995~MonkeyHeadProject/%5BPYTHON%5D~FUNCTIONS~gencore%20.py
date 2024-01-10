#!/usr/bin/env python3

import os
import sys
import datetime
import configparser
import openai
import platform
import logging 
import subprocess
import socket
import psutil
import tempfile
import json
import requests
from bs4 import BeautifulSoup
from contextlib import contextmanager

from CLI.cli_print import print_message

logging.basicConfig(filename='gizmo_record.log', level=logging.DEBUG)


# Define CoreGizmo class.
class CoreGizmo:
    def __init__(self, gizmo_code, default_config):
        self.current_folder = os.getcwd()
        self.parent_folder = os.path.dirname(self.current_folder)
        self.volume_names = {
            "GIZMO": [],
            "MEMORY": [],
            "BOOKS": [],
            "CONFIG": ["config_manager.py"],
            "CLI": ["gizmo_cli.py"],
            "GUI": ["gizmo_gui.py"],
            "SPLITTER": ["text_splitter.py"],
            "FORMATTER": ["text_formatter.py"],
            "MONOPOLY": ["monopoly_simulator.py"],
            "BananaBrain": ["banana_brain.py"],
            "AGREEMENT": []
        }
        default_config = {
            "General": {
                "Encoding": "utf-8",
                "MaxRetries": 3,
                "Temperature": 0.2,
                "TokenLimit": 8192
            },
            "Paths": {
                "MemoryFile": "MEMORY/Core_Memory.txt",
                "ConfigFile": "CONFIG/Config_Settings.ini",
                "MemoryFilePath": "MEMORY/Core_History"
            },
            "Libraries": [
                "os", "sys", "datetime", "configparser", "openai", "platform",
                "logging", "subprocess", "socket", "psutil", "tempfile",
                "requests", "bs4.BeautifulSoup", "contextlib"
            ],
            "MessageTypes": [
                "initialization", "status", "info", "error", "debug", "alert",
                "user_input", "query", "summary", "warning", "fatal_error",
                "shutdown"
            ],
            "API": {
                "APIKey": "Your-API-Key-Here",
                "GPTModel": ["gpt-4", "gpt-3.5-turbo"]
            },
            "Role": {
                "RoleDescription": "Your name is Gizmo, and you're an AI Lab Assistant..."
            }
        }
        self.default_config = default_config
        self.buffer = []
        self.load_config_and_memory()
        self.setup_environment()
        self.initial_message()
        
    def graceful_exit(self):
        self.print_message("Cleaning up resources...", "status")
        self.print_message("Exiting the program.", "status")
        sys.exit()

    def initial_message(self):
        program_name = os.path.basename(__file__).replace('.py', '')
        formatted_program_name = ' '.join(word.capitalize() for word in program_name.split('_'))
        
        title_message = f"[Initialization] Monkey Head Project: {formatted_program_name} Initialized"
        init_message = f"Starting '{formatted_program_name}.py'..."
        
        common_prefix = "Monkey Head Project: Gizmo / BananaBrain"
        
        print(f"{common_prefix} - {formatted_program_name}")
        self.print_message(title_message, "initialization")
        self.print_message(init_message, "initialization")


    

    def setup_folders_and_files(self):
        ensure_folder_exists(os.path.join(self.parent_folder, self.level_1))

        for folder, py_files in self.level_2_folders.items():
            folder_path = os.path.join(self.parent_folder, self.level_1, folder)
            self.ensure_folder_exists(folder_path)

            for py_file in py_files:
                self.create_python_file(folder_path, py_file, python_code.get(py_file, "# Sample code"))

      

    def create_python_file(self, folder_path, py_file_name, code_content):
        py_file_path = os.path.join(folder_path, py_file_name)
        if not os.path.exists(py_file_path):
            with open(py_file_path, 'w') as f:
                f.write(code_content)
        
    @contextmanager
    def load_config(self, config_path=None):
        if config_path is None:
            config_path = self.config_file
        config = configparser.ConfigParser()
        config.read(config_path)
        self.validate_config(config)
        self.validate_model(config)
        try:
            yield config
        finally:
            with open(config_path, "w") as configfile:
                config.write(configfile)

    def reload_config(self):
        with self.load_config(self.config_file) as config:
            self.api_key = self.get_api_key(config)
            role_description, model, temperature, file_path = self.get_settings(config)

    def get_api_key(self, config):
        api_key = config.get('openai', 'api_key', fallback=os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            self.print_message("API key is missing. Please add it to the config file or set the OPENAI_API_KEY environment variable.", "error")
            self.graceful_exit()
        return api_key

    def get_settings(self, config):
        if 'settings' not in config.sections():
            self.print_message("Missing 'settings' section. Add it with the necessary keys.", "error")
            raise ValueError()
        default_role_description = config.get("settings", "default_role_description")
        default_model = config.get("settings", "default_model")
        default_temperature = config.getfloat("settings", "default_temperature")
        file_path = config.get("settings", "file_path")
        return default_role_description, default_model, default_temperature, file_path

    def validate_question(self, question: str):
        if not question.strip():
            self.print_message("The question cannot be empty. Please enter a valid question.", "error")
            raise ValueError()
        if len(question.split()) > self.token_limit:
            self.print_message("The question is too long. Please enter a question with fewer tokens.", "error")
            raise ValueError()

    def validate_model(self, model: str):
        if model not in self.gpt_model:
            self.print_message(f"The model '{model}' is not valid. Please enter a valid model.", "error")
            raise ValueError()

    

    def get_user_input(self):
        return self.print_message(f"INPUT: ", "user_input")

    def get_openai_answer(self):
        input_content = self.get_user_input()
        openai.api_key = self.api_key
        if isinstance(input_content, list): 
            messages = input_content
        else:  
            messages = [{"role": "system", "content": "You're an AI Assistant."}, {"role": "user", "content": input_content}]
        response = openai.Completion.create(
            engine=self.gpt_model[0],
            prompt=input_content,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    
    def add_to_buffer(self, role: str, content: str):
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        memory_entry = f"{timestamp} - {role}: {content}"
        
        self.buffer.append(memory_entry)
        self.save_to_memory_file(memory_entry)

    def get_history(self):
        return self.buffer

    def save_to_memory_file(self, memory_entry: str):
        with open(self.memory_file, "a") as file:
            file.write(f"{memory_entry}\n")
        self.print_message(f"Saved entry to memory file: {memory_entry}", "status")

    def load_from_memory_file(self):
        try:
            with open(self.memory_file, "r") as file:
                self.buffer = file.readlines()
            self.buffer = [line.strip() for line in self.buffer]
        except FileNotFoundError:
            self.print_message("Memory file not found. Creating a new one.", "warning")
            self.buffer = []

    def generate_answer(self, question: str):
        self.validate_question(question)

        self.add_to_buffer("user", question)
        conversation_history = self.get_history()

        with self.load_config(self.config_file) as config:
            api_key = self.get_api_key(config)
            model = self.gpt_model
            temperature = self.temperature
            
        answer = self.get_openai_answer(conversation_history, api_key, model, temperature)

        self.add_to_buffer("assistant", answer)
        return answer
            
    def main(self):
        try:
            self.ensure_folders_exist()

            with self.load_config() as config:
                api_key = self.get_api_key(config)
                role_description, model, temperature, file_path = self.get_settings(config)
                self.add_to_buffer("system", role_description)
                
                while True:
                    question = self.print_message("Enter a question or type 'quit' to exit or 'reload config' to reload the configuration", "user_input").strip()
                    
                    if question.lower() == 'quit':
                        self.graceful_exit()
                        break
                    elif question.lower() == 'reload config':
                        self.reload_config()
                        self.print_message("Configuration reloaded.", "status")
                        continue

                    self.validate_question(question)
                    answer = self.generate_answer(question)
                    
                    self.print_message(answer, "info")
                    self.add_to_buffer("user", question)
                    self.add_to_buffer("assistant", answer)
        except Exception as e:
            self.print_message(f"An exception occurred: {e}", "error")
            self.handle_exception(type(e), "main")
            self.graceful_exit()


if __name__ == '__main__':
    pyhton_code = {
        "config_manager.py": "# Code to handle custom config files",
        "gizmo_cli.py": "# Code for CLI communication",
        "gizmo_gui.py": "# Code for GUI communication",
        "text_splitter.py": "# Code to split books into pieces",
        "text_formatter.py": "# Code to format book information",
        "monopoly_simulator.py": "# Code for Monopoly simulator with GPT interface",
        "banana_brain.py": "# Code for neural network related tasks"
    }
    core = CoreGizmo(python_code, default_config)
    core.main()
