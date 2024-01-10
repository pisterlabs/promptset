#!/usr/bin/python
import collections
import os
import ctypes
import json
import queue
import sys
import threading
import time

import configparser
import random
import traceback

import irc.bot
import requests
import argparse

import openai
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from tkinter import messagebox, ttk, font, IntVar
import tkinter.scrolledtext as tkscrolled
import tkinter as tk

from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser
# import websocket
import gpt4all
import io
from contextlib import redirect_stdout
from html import escape


def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def get_version():
    return "1.66"  # Version Number


class TwitchBotGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("pyWiki Lite")
        self.geometry("1000x425")
        self.iconbitmap(default=resource_path('icon.ico'))

        # Make the window non-resizable
        self.resizable(False, False)

        # Variables for Twitch Bot configuration
        self.username = tk.StringVar()
        self.client_id = tk.StringVar()
        self.client_secret = tk.StringVar()
        self.bot_token = tk.StringVar()
        self.refresh_token = tk.StringVar()
        self.channel = tk.StringVar()
        self.openai_api_key = tk.StringVar()
        self.openai_api_model = tk.StringVar()
        self.ignore_userlist = IntVar()

        # Variable to keep track of the bot state
        self.bot_running = False

        self.mute = False

        self.openai_models = ['gpt-4', 'gpt-3.5-turbo']
        if os.path.exists('ggml-mpt-7b-chat.bin'):
            self.openai_models.append('mpt-7b-chat')
        if os.path.exists('wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin'):
            self.openai_models.append('WizardLM-13B')

        self.create_widgets()

        # Load configuration from the INI file
        if not os.path.exists('config.ini'):
            self.save_configuration()
        self.load_configuration()

        # Bind the on_exit function to the closing event of the Tkinter window
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Initialize a Queue for handling log messages
        self.log_queue = queue.Queue()

        # Start a separate thread to update the log asynchronously
        self.log_thread = threading.Thread(target=self.process_log_queue)
        self.log_thread.daemon = True
        self.log_thread.start()

    # Function to handle selection change
    def on_selection_change(self, event):
        self.openai_api_model.set(self.openai_model_entry.get())
        print(self.openai_model_entry.get() + ' set')
        self.append_to_log(self.openai_model_entry.get() + ' set')

    def show_about_popup(self):
        about_text = "pyWiki Lite " + get_version() + "\n©2023 Ixitxachitl\nAnd ChatGPT"
        thread = threading.Thread(target=lambda: messagebox.showinfo("About", about_text))
        thread.start()

    def append_to_log(self, message):
        self.log_queue.put(message)

    def process_log_queue(self):
        while True:
            try:
                message = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)  # Enable the Text widget for editing
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)  # Scroll to the bottom of the text widget
                self.log_text.config(state=tk.DISABLED)  # Disable the Text widget for editing
            except queue.Empty:
                pass
            time.sleep(.1)

    def toggle_stay_on_top(self):
        if self.attributes("-topmost"):
            self.attributes("-topmost", False)
            self.stay_on_top_button.config(relief="raised")
        else:
            self.attributes("-topmost", True)
            self.stay_on_top_button.config(relief="sunken")

    def toggle_mute(self):
        if self.mute == True:
            self.mute = False
            self.append_to_log('Unmuted')
            self.stay_mute_button.config(relief="raised")
        else:
            self.mute = True
            self.append_to_log('Muted')
            self.stay_mute_button.config(relief="sunken")

    def create_widgets(self):
        # Set the column weight to make text inputs expand horizontally
        self.columnconfigure(1, weight=1)

        tk.Label(self, text="pyWiki Lite Configuration", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2,
                                                                                      pady=10, padx=10, sticky='w')
        tk.Label(self, text="Context", font=("Helvetica", 16)).grid(row=0, column=3, columnspan=1,
                                                                    pady=0, padx=(0, 10), sticky='w')
        tk.Label(self, text="Users", font=("Helvetica", 16)).grid(row=0, column=5, columnspan=1, padx=(0, 10),
                                                                  sticky='w')
        self.user_count = tk.Label(self, text="", font=("Helvetica 16 bold"))
        self.user_count.grid(row=0, column=6, columnspan=1, pady=10, padx=(0, 10), sticky='w')

        # Twitch Bot Username Entry
        tk.Label(self, text="Username:").grid(row=1, column=0, padx=(10, 5), sticky="e")
        '''
        self.bot_username_entry = tk.Entry(self, textvariable=self.username, width=50)
        self.bot_username_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(2, 0))
        '''
        self.bot_username_entry = tk.Label(self, textvariable=self.username)
        self.bot_username_entry.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=(2, 0))

        self.login_button = tk.Button(self, text="Login", command=self.twitch_login)
        self.login_button.grid(row=1, column=1, sticky="e", padx=10)

        # ClientID Entry
        tk.Label(self, text="ClientID:").grid(row=2, column=0, padx=(10, 5), sticky="e")
        self.client_id_entry = tk.Entry(self, show="*", textvariable=self.client_id, width=50)
        self.client_id_entry.grid(row=2, column=1, sticky="ew", padx=(0, 10))

        # Client Secret Entry
        tk.Label(self, text="Client Secret:").grid(row=3, column=0, padx=(10, 5), sticky="e")
        self.client_secret_entry = tk.Entry(self, show="*", textvariable=self.client_secret, width=50)
        self.client_secret_entry.grid(row=3, column=1, sticky="ew", padx=(0, 10))

        '''
        # Twitch Bot Token Entry
        tk.Label(self, text="Bot OAuth Token:").grid(row=4, column=0, padx=(10, 5), sticky="e")
        self.bot_token_entry = tk.Entry(self, show="*", textvariable=self.bot_token, width=50)
        self.bot_token_entry.grid(row=4, column=1, sticky="ew", padx=(0, 10))
        '''

        # Channel Entry
        tk.Label(self, text="Channel:").grid(row=4, column=0, padx=(10, 5), sticky="e")
        self.channel_entry = tk.Entry(self, textvariable=self.channel, width=50)
        self.channel_entry.grid(row=4, column=1, sticky="ew", padx=(0, 10))

        # OpenAI API Key Entry
        tk.Label(self, text="OpenAI API Key:").grid(row=5, column=0, padx=(10, 5), sticky="e")
        self.openai_api_key_entry = tk.Entry(self, show="*", textvariable=self.openai_api_key, width=50)
        self.openai_api_key_entry.grid(row=5, column=1, sticky="ew", padx=(0, 10))

        # OpenAI Model
        self.openai_model_entry = ttk.Combobox(self, textvariable=self.openai_api_model, state="readonly")
        self.openai_model_entry['values'] = self.openai_models
        self.openai_model_entry.grid(row=0, column=4, sticky="e", padx=10)

        # Set the default value for the dropdown box
        self.openai_model_entry.current(0)

        # Bind the event handler to the selection change event
        self.openai_model_entry.bind('<<ComboboxSelected>>', self.on_selection_change)

        self.stay_on_top_button = tk.Button(self, text="📌", command=self.toggle_stay_on_top)
        self.stay_on_top_button.grid(row=0, column=7, sticky="e", padx=10)

        self.about_button = tk.Button(self, text="ℹ️", command=self.show_about_popup, borderwidth=0)
        self.about_button.grid(row=0, column=7, columnspan=2, sticky="e")

        self.stay_mute_button = tk.Button(self, text="🔇", font=font.Font(size=14), justify='center',
                                          command=self.toggle_mute)
        self.stay_mute_button.grid(row=6, column=0, columnspan=2, sticky="e", padx=(0, 10))

        # Create a slider widget
        self.frequency_slider = tk.Scale(self, from_=0, to=100, orient=tk.HORIZONTAL)
        self.frequency_slider.grid(row=6, column=0, columnspan=2, padx=(10, 60), pady=0, sticky="ew")

        self.frequency_slider.bind("<Enter>", self.on_frequency_slider_enter)
        self.frequency_slider.bind("<Leave>", self.on_frequency_slider_leave)

        # Start/Stop Bot Button
        self.bot_toggle_button = tk.Button(self, text="Start Bot", command=self.toggle_bot)
        self.bot_toggle_button.grid(row=0, column=1, columnspan=1, sticky="e", pady=10, padx=10)

        # Create a Text widget to display bot messages
        self.log_text = tkscrolled.ScrolledText(self, wrap="word", height=11, state=tk.DISABLED)
        self.log_text.grid(row=7, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="ewn")

        # Create a Text widget to display the input string
        self.input_text = tkscrolled.ScrolledText(self, wrap="word", height=22, width=40, undo=True,
                                                  autoseparators=True, maxundo=-1)
        self.input_text.grid(row=1, column=3, columnspan=2, rowspan=7, padx=(0, 10), pady=(10, 0), sticky="ne")

        # Create a Listbox to display users
        self.ignore_userlist_check = tk.Checkbutton(self, text="Ignore User List", variable=self.ignore_userlist,
                                                    onvalue=1,
                                                    offvalue=0)
        self.ignore_userlist_check.grid(row=1, column=5, columnspan=3, sticky='nw', pady=0)
        self.user_list_scroll = tk.Scrollbar(self, orient="vertical")
        self.user_list_scroll.grid(row=2, column=8, columnspan=1, rowspan=6, pady=0, padx=(0, 10), sticky="ns")
        self.user_list = tk.Listbox(self, height=21, selectmode='SINGLE', width=30,
                                    yscrollcommand=self.user_list_scroll.set)
        self.user_list_scroll.config(command=self.user_list.yview)
        self.user_list.grid(row=2, column=5, columnspan=3, rowspan=6, pady=0, sticky="ne")
        self.user_list.bind('<FocusOut>', lambda e: self.user_list.selection_clear(0, tk.END))
        self.user_list.bind('<Double-Button-1>', self.show_popup)
        self.user_list.bind('<Button-3>', self.message_user)

    def on_frequency_slider_enter(self, event):
        self.frequency_slider.bind("<MouseWheel>", self.on_frequency_slider_scroll)

    def on_frequency_slider_leave(self, event):
        self.frequency_slider.unbind("<MouseWheel>")

    def on_frequency_slider_scroll(self, event):
        current_value = self.frequency_slider.get()
        if event.delta > 0:
            new_value = min(current_value + 1, self.frequency_slider['to'])
        else:
            new_value = max(current_value - 1, self.frequency_slider['from'])
        self.frequency_slider.set(new_value)

    def message_user(self, event):
        selected_index = self.user_list.curselection()
        if selected_index:
            item_index = int(selected_index[0])
            selected_item = self.user_list.get(item_index)
            if selected_item.lower() in self.bot.last_message.keys():
                thread = threading.Thread(
                    target=lambda: self.bot.generate_response(selected_item,
                                                              self.bot.last_message[selected_item.lower()]))
            else:
                thread = threading.Thread(
                    target=lambda: self.bot.generate_response(selected_item, '@ ' + selected_item))
            thread.start()

    def show_popup(self, event):
        selected_index = self.user_list.curselection()
        if selected_index:
            item_index = int(selected_index[0])
            selected_item = self.user_list.get(item_index)
            url = 'https://api.twitch.tv/helix/users?login=' + selected_item
            headers = {
                'Authorization': 'Bearer ' + self.bot_token.get(),
                'Client-Id': self.client_id.get(),
                'Content-Type': 'application/json',
            }
            while True:
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    break
                elif response.status_code == 401:
                    self.refresh_login()
                else:
                    # Handle other status codes if needed
                    messagebox.showerror("Error", "Error fetching data: " + str(response.status_code))
                    return

            # Now you can safely access the data from the response
            try:
                created_at = response.json()['data'][0]['created_at']
                followed_at = self.bot.get_followage(selected_item)

                try:
                    con_followed_at = datetime.strptime(followed_at, '%Y-%m-%dT%H:%M:%SZ')
                    follow_time = relativedelta(datetime.utcnow(), con_followed_at)

                    time_units = [('year', follow_time.years), ('month', follow_time.months), ('day', follow_time.days),
                                  ('hour', follow_time.hours)]
                    time_strings = [f"{value} {unit}" if value == 1 else f"{value} {unit}s" for unit, value in
                                    time_units if
                                    value > 0]
                    time_string = ', '.join(time_strings)
                except ValueError:
                    time_string = ''

                thread = threading.Thread(target=lambda: messagebox.showinfo(selected_item, 'Created on: ' + created_at
                                                                             + '\nFollowed on: ' + followed_at + '\n' +
                                                                             time_string))
                thread.start()
            except KeyError:
                messagebox.showerror("Error", "Error parsing response data")
            except IndexError:
                messagebox.showerror("Error", "Missing response data")

    def twitch_login(self):
        self.open_browser_and_start_server()

    def refresh_login(self):
        auth_params = {
            'client_id': self.client_id.get(),
            'client_secret': self.client_secret.get(),
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token.get(),
        }
        response = requests.post('https://id.twitch.tv/oauth2/token', data=auth_params)
        data = response.json()
        self.bot_token.set(data['access_token'])
        self.refresh_token.set(data['refresh_token'])

    def open_browser_and_start_server(self):
        print('Logging in...')
        print(self.client_id.get())
        # Open the authorization URL in the default web browser
        auth_params = {
            'client_id': self.client_id.get(),
            'redirect_uri': 'http://localhost:3000',
            'response_type': 'code',
            'scope': 'chat:read+chat:edit+channel:moderate+whispers:read+whispers:edit+channel_editor+user:read:follows+moderator:read:followers+channel:read:redemptions',
            'force_verify': 'true',
        }
        auth_url = 'https://id.twitch.tv/oauth2/authorize?' + '&'.join([f'{k}={v}' for k, v in auth_params.items()])
        webbrowser.open(auth_url)

        # Start the server in a separate thread
        server_address = ('', 3000)
        httpd = HTTPServer(server_address, CallbackHandler)
        httpd.handle_request()

    def write_to_text_file(self, file_path, content):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"Successfully wrote to {file_path}")
        except Exception as e:
            print(f"Error occurred while writing to {file_path}: {e}")

    def load_configuration(self):
        config = configparser.ConfigParser()
        if not config.read('config.ini'):
            return

        section = config['TwitchBot']
        self.username.set(section.get('username', ''))
        self.client_id.set(section.get('ClientID', ''))
        self.client_secret.set(section.get('ClientSecret', ''))
        self.bot_token.set(section.get('BotOAuthToken', ''))
        self.refresh_token.set(section.get('RefreshToken', ''))
        self.channel.set(section.get('InitialChannels', ''))
        self.openai_api_key.set(section.get('OpenAIAPIKey', ''))
        if not section.get('InputString', ''):
            self.input_text.insert(tk.END, "You are a twitch chatbot, your username is <name> and your pronouns "
                                           "are They/Them. The name of the streamer is <channel> and their "
                                           "pronouns are <streamer_pronouns>. The streamer is playing <game>. The "
                                           "name of the chatter is <author> and their pronouns are "
                                           "<chatter_pronouns>. The current date and time are: <time>. A list of "
                                           "users in chat are: <users>. Global twitch emotes that you can use are"
                                           " <emotes>.")
        else:
            self.input_text.insert(tk.END, section.get('InputString', ''))

        if not section.get('Model', ''):
            self.openai_api_model.set('gpt-4-0613')
        else:
            self.openai_api_model.set(section.get('Model', ''))

        if not section.get('Frequency', ''):
            self.frequency_slider.set(0)
        else:
            self.frequency_slider.set(section.get('Frequency', ''))

        self.ignore_userlist.set(int(section.get('IgnoreUsers', '0')))

    def save_configuration(self):
        config = configparser.ConfigParser()
        config['TwitchBot'] = {
            'username': self.username.get(),
            'ClientID': self.client_id.get(),
            'ClientSecret': self.client_secret.get(),
            'BotOAuthToken': self.bot_token.get(),
            'RefreshToken': self.refresh_token.get(),
            'InitialChannels': self.channel.get(),
            'OpenAIAPIKey': self.openai_api_key.get(),
            'InputString': self.input_text.get('1.0', 'end'),
            'Model': self.openai_api_model.get(),
            'Frequency': self.frequency_slider.get(),
            'IgnoreUsers': self.ignore_userlist.get()
        }

        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    def start_bot(self):
        if not self.bot_running:
            self.refresh_login()
            self.bot_running = True
            self.bot_toggle_button.config(text="Stop Bot")

            # Start the bot in a separate thread
            self.bot_thread = threading.Thread(target=self.run_bot, daemon=False)
            self.bot_thread.start()
            return

    def run_bot(self):
        # This method will be executed in a separate thread
        # Create and run the bot here
        self.bot = TwitchBot(self.username.get(), self.client_id.get(), self.client_secret.get(), self.bot_token.get(),
                             self.channel.get(), self.openai_api_key.get())
        self.bot.start()
        return

    def stop_bot(self):
        if self.bot_running:
            self.bot_running = False
            self.bot_toggle_button.config(text="Start Bot")
            self.write_to_text_file("log.txt", self.log_text.get("1.0", tk.END).strip())
            self.user_list.delete(0, tk.END)
            app.user_count.config(text="")
            if hasattr(self, "bot"):
                try:
                    self.bot.connection.quit()
                    self.bot.disconnect()
                except Exception as e:
                    print(e)
                # self.bot.die()
                # self.bot_thread.join()
                self.terminate_thread(self.bot_thread)
                self.bot.users = []
                print("Stopped")
                self.append_to_log("Stopped")

    def terminate_thread(self, thread):
        """Terminates a python thread from another thread.

        :param thread: a threading.Thread instance
        """
        if not thread.is_alive():
            return

        exc = ctypes.py_object(SystemExit)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread.ident), exc)
        if res == 0:
            raise ValueError("nonexistent thread id")
        elif res > 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def on_exit(self):
        self.save_configuration()
        self.stop_bot()
        self.destroy()

    def toggle_bot(self):
        self.save_configuration()
        if self.bot_running:
            self.bot_toggle_button.config(relief="raised")
            self.login_button.config(state=tk.NORMAL)
            self.client_id_entry.config(state="normal")
            self.client_secret_entry.config(state="normal")
            self.channel_entry.config(state="normal")
            self.openai_api_key_entry.config(state="normal")
            self.stop_bot()
        else:
            self.bot_toggle_button.config(relief="sunken")
            self.login_button.config(state=tk.DISABLED)
            self.client_id_entry.config(state="disabled")
            self.client_secret_entry.config(state="disabled")
            self.channel_entry.config(state="disabled")
            self.openai_api_key_entry.config(state="disabled")
            self.start_bot()


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Extract access token from the URL fragment
        fragment = self.path.split('?')[1]
        fragment_params = dict(param.split('=') for param in fragment.split('&'))
        code = fragment_params.get('code')

        token_params = {
            'client_id': app.client_id.get(),
            'client_secret': app.client_secret.get(),
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': 'http://localhost:3000',
        }

        response = requests.post('https://id.twitch.tv/oauth2/token', data=token_params)
        data = response.json()
        access_token = data['access_token']
        refresh_token = data['refresh_token']
        print('Access Token: ' + access_token)
        print('Refresh Token: ' + refresh_token)

        url = 'https://api.twitch.tv/helix/users'
        headers = {'Authorization': 'Bearer ' + access_token,
                   'Client-ID': app.client_id.get(),
                   'Content-Type': 'application/json'}
        response = requests.get(url, headers=headers).json()
        username = response['data'][0]['login']
        print('Login: ' + username)

        app.bot_token.set(access_token)
        app.refresh_token.set(refresh_token)
        app.username.set(username)

        # Now you can use the access_token to make authenticated API requests
        self.wfile.write(b'Authorization successful! You can close this window now.')


class TwitchBot(irc.bot.SingleServerIRCBot):
    def __init__(self, username, client_id, client_secret, token, channel, openai_api_key):
        self.username = username
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = token
        self.channel = '#' + channel

        self.client_credentials = requests.post('https://id.twitch.tv/oauth2/token?client_id='
                                                + self.client_id
                                                + '&client_secret='
                                                + self.client_secret
                                                + '&grant_type=client_credentials'
                                                + '').json()

        print(self.client_credentials)
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key

        self.pronoun_cache = {}
        self.users = []
        self.message_queue = collections.deque(maxlen=10)
        self.last_message = {}

        self.verify()
        self.channel_id = self.get_channel_id(channel)
        self.user_id = self.get_channel_id(username)
        self.emotes = self.get_emotes()

        self.functions = [
            # {
            #    "name": "get_user_pronouns",
            #    "description": "Get the pronouns of a specified user",
            #    "parameters": {
            #        "type": "object",
            #        "properties": {
            #            "user": {
            #                "type": "string",
            #                "description": "The name of the person to look up pronouns for",
            #            },
            #        },
            #        "required": ["user"],
            #    },
            # },
            {
                "name": "get_launch",
                "description": "Get the next or previous scheduled space launch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "when": {
                            "type": "string",
                            "enum": ["next", "previous"]
                        },
                    },
                    "required": ["when"],
                },
            },
            {
                "name": "get_users",
                "description": "Get a list of users in chat",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                },
            },
            {
                "name": "get_stream",
                "description": "Gets information about a stream by streamer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "streamer": {
                            "type": "string",
                            "description": "the name of the streamer to look up"
                        },
                    },
                    "required": ["streamer"],
                },
            },
            {
                "name": "get_game_info",
                "description": "Gets information about a game",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game": {
                            "type": "string",
                            "description": "the name of the game to look up"
                        },
                    },
                    "required": ["game"],
                },
            },
            {
                "name": "send_message_delayed",
                "description": "sends a message after a number of seconds",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "the message to send"
                        },
                        "delay_seconds": {
                            "type": "string",
                            "description": "the number of seconds to delay"
                        },
                    },
                    "required": ["message", "delay_seconds"],
                },
            },
        ]

        # Create IRC bot connection
        server = 'irc.chat.twitch.tv'
        port = 6667
        print('Connecting to ' + server + ' on port ' + str(port) + '...')
        app.append_to_log('Connecting to ' + server + ' on port ' + str(port) + '...')
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port, 'oauth:' + token)], username, username)

    '''
    def receive_twitch_events(self):
        twitch_uri = "wss://eventsub.wss.twitch.tv/ws"

        def on_message(ws, message):
            data = json.loads(message)
            print(data)

            if data['metadata']['message_type'] == "PING":
                # Respond with a pong
                ws.send(json.dumps({"type": "PONG"}))
                print('PONG')
            elif data['metadata']['message_type'] == "session_welcome":
                session_id = data['payload']['session']['id']
                headers = {
                    'Authorization': 'Bearer ' + app.bot_token.get(),
                    'Client-Id': app.client_id.get(),
                    'Content-Type': 'application/json',
                }
                auth_params = {
                    'type': 'channel.channel_points_custom_reward_redemption.add',
                    'version': '1',
                    'condition': {"broadcaster_user_id": self.channel_id},
                    'transport': {"method": "websocket", "session_id": session_id},
                }
                response = requests.post('https://api.twitch.tv/helix/eventsub/subscriptions', json=auth_params, headers=headers)
                print(response.json())

        ws = websocket.WebSocketApp(twitch_uri, on_message=on_message)
        ws.run_forever()
        '''

    def verify(self):
        url = 'https://id.twitch.tv/oauth2/validate'
        headers = {'Authorization': 'OAuth ' + app.bot_token.get()}
        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                break
            elif response.status_code == 401:
                app.refresh_login()
            else:
                # Handle other status codes if needed
                return "Error fetching data: " + str(response.status_code)

        # Now you can safely access the data from the response
        try:
            verification = response.json()
            return verification
        except KeyError:
            return "Error parsing response data"
        except IndexError:
            return "Missing response data"

    def get_channel_id(self, channel, **kwargs):
        # Get the channel id, we will need this for v5 API calls
        print('Called get_channel_id for ' + channel)
        app.append_to_log('Called get_channel_id for ' + channel)
        url = 'https://api.twitch.tv/helix/users?login=' + escape(channel)
        headers = {
            'Authorization': 'Bearer ' + app.bot_token.get(),
            'Client-Id': app.client_id.get(),
            'Content-Type': 'application/json',
        }

        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                break
            elif response.status_code == 401:
                app.refresh_login()
            else:
                # Handle other status codes if needed
                return "Error fetching data: " + str(response.status_code)

        # Now you can safely access the data from the response
        try:
            channel_id = response.json()['data'][0]['id']
            return channel_id
        except KeyError:
            return "Error parsing response data"
        except IndexError:
            return "Missing response data"

    def get_game(self, channel, **kwargs):
        print('Called get_game for ' + channel)
        app.append_to_log('Called get_game for ' + channel)
        # Get the current game
        url = 'https://api.twitch.tv/helix/channels?broadcaster_id=' + escape(channel)
        headers = {
            'Authorization': 'Bearer ' + app.bot_token.get(),
            'Client-Id': app.client_id.get(),
            'Content-Type': 'application/json',
        }
        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                break
            elif response.status_code == 401:
                app.refresh_login()
            else:
                # Handle other status codes if needed
                return "Error fetching data: " + str(response.status_code) + " " + str(response.content)

        # Now you can safely access the data from the response
        try:
            game_name = response.json()['data'][0]['game_name']
            return game_name
        except KeyError:
            return "Error parsing response data"
        except IndexError:
            return "Missing response data"

    def get_game_info(self, game, **kwargs):
        print('Called get_game_info for ' + game)
        app.append_to_log('Called get_game_info for ' + game)
        url = 'https://api.igdb.com/v4/games'
        headers = {
            'Authorization': 'Bearer ' + self.client_credentials['access_token'],
            'Client-Id': app.client_id.get(),
            'Content-Type': 'application/json',
        }
        data = 'fields *; where name ~ "' + escape(game) + '";'
        print(data)
        response = requests.post(url, headers=headers, data=data)
        print(response)
        game_info = json.dumps(response.json())
        print(game_info)
        return game_info

    def get_emotes(self, **kwargs):
        # Get list of global emotes
        print('Called get_emotes')
        app.append_to_log('Called get_emotes')
        url = 'https://api.twitch.tv/helix/chat/emotes/global'
        headers = {
            'Authorization': 'Bearer ' + app.bot_token.get(),
            'Client-Id': app.client_id.get(),
            'Content-Type': 'application/json',
        }
        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                break
            elif response.status_code == 401:
                app.refresh_login()
            else:
                # Handle other status codes if needed
                return "Error fetching data: " + str(response.status_code)

        # Now you can safely access the data from the response
        try:
            emotes = []
            for emote in response.json()['data']:
                emotes.append(emote['name'])
            return emotes
        except KeyError:
            return "Error parsing response data"
        except IndexError:
            return "Missing response data"

    def get_stream(self, streamer, **kwargs):
        print('Called get_stream for ' + streamer)
        app.append_to_log('Called get_stream for ' + streamer)
        if streamer == None:
            streamer = self.channel[1:]
        url = 'https://api.twitch.tv/helix/search/channels?query=' + escape(streamer) + '&first=1'
        headers = {
            'Authorization': 'Bearer ' + app.bot_token.get(),
            'Client-Id': app.client_id.get(),
            'Content-Type': 'application/json',
        }
        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                break
            elif response.status_code == 401:
                app.refresh_login()
            else:
                # Handle other status codes if needed
                return "Error fetching data: " + str(response.status_code)

        # Now you can safely access the data from the response
        try:
            print(response.json())
            stream = json.dumps(response.json()['data'][0])
            return stream
        except KeyError:
            return "Error parsing response data"
        except IndexError:
            return "Missing response data"

    def get_followage(self, user, **kwargs):
        print('Called get_followage for ' + user)
        app.append_to_log('Called get_followage for ' + user)

        headers = {'Authorization': 'Bearer ' + app.bot_token.get(),
                   'Client-ID': app.client_id.get(),
                   'Content-Type': 'application/json'}
        url = 'https://api.twitch.tv/helix/channels/followers?user_id=' + escape(self.get_channel_id(
            user)) + '&broadcaster_id=' + escape(self.channel_id)

        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                break
            elif response.status_code == 401:
                app.refresh_login()
            else:
                # Handle other status codes if needed
                return "Error fetching data: " + str(response.json()['message'])

        # Now you can safely access the data from the response
        try:
            followage = response.json()['data'][0]['followed_at']
            return followage
        except KeyError:
            return "Error parsing response data"
        except IndexError:
            return "Not Following"

    def get_users(self, **kwargs):
        print('Called get_users')
        app.append_to_log('Called get_users')
        self.connection.users()
        if app.ignore_userlist.get() == 1:
            return 'unknown'
        else:
            return str(app.user_list.get(0, tk.END))

    def on_namreply(self, c, e):
        for key in self.channels[self.channel].users():
            if key not in self.users:
                self.users.append(key)

        for user in self.users:
            if user not in str(app.user_list.get(0, tk.END)):
                app.user_list.insert(tk.END, user)

        app.user_count.config(text=app.user_list.size())

        print(', '.join(map(str, self.users)))

    def on_join(self, c, e):
        user = e.source.nick
        if user not in str(app.user_list.get(0, tk.END)):
            self.users.append(user)
            app.user_list.insert(tk.END, user)
            app.user_count.config(text=app.user_list.size())
            print(user + ' joined')

    def on_part(self, c, e):
        user = e.source.nick
        if user in str(self.users):
            self.users.remove(user)
            for index in range(app.user_list.size(), -1, -1):
                item_name = app.user_list.get(index)
                if item_name == user:
                    app.user_list.delete(index)
            app.user_count.config(text=app.user_list.size())
            print(user + ' left')

    def on_welcome(self, c, e):
        print('Joining ' + self.channel)
        app.append_to_log('Joining ' + self.channel)
        self.connection = c

        # You must request specific capabilities before you can use them
        c.cap('REQ', ':twitch.tv/membership')
        c.cap('REQ', ':twitch.tv/tags')
        c.cap('REQ', ':twitch.tv/commands')
        c.join(self.channel)

        self.connection.users()

        '''
        thread = threading.Thread(target=self.receive_twitch_events)
        thread.start()
        thread.join()
        '''

    def get_launch(self, when, **kwargs):
        print('Called get_launch on ' + when)
        app.append_to_log('Called get_launch on ' + when)
        if when == 'next':
            url = 'https://ll.thespacedevs.com/2.2.0/launch/upcoming/?mode=list'
        else:
            url = 'https://ll.thespacedevs.com/2.2.0/launch/previous/?mode=list'
        return json.dumps(requests.get(url).json()["results"][:2])

    def get_pronouns(self, author, **kwargs):
        print('Called get_pronouns for ' + author)
        app.append_to_log('Called get_pronouns for ' + author)
        # Check if pronouns exist in the cache
        if author.lower() in self.pronoun_cache:
            return self.pronoun_cache[author.lower()]

        url = 'https://pronouns.alejo.io/api/users/' + escape(author.lower())
        r = requests.get(url).json()

        pronoun_mapping = {
            'aeaer': 'Ae/Aer',
            'any': 'Any',
            'eem': 'E/Em',
            'faefaer': 'Fae/Faer',
            'hehim': 'He/Him',
            'heshe': 'He/She',
            'hethem': 'He/They',
            'itits': 'It/Its',
            'other': 'Other',
            'perper': 'Per/Per',
            'sheher': 'She/Her',
            'shethey': 'She/They',
            'theythem': 'They/Them',
            'vever': 'Ve/Ver',
            'xexem': 'Xe/Xem',
            'ziehir': 'Zie/Hir'
        }

        pronouns = r[0]['pronoun_id'] if r else 'unknown'
        pronoun = pronoun_mapping.get(pronouns, 'unknown')

        print('Got ' + author + ' pronouns ' + pronoun)
        app.append_to_log('Got ' + author + ' pronouns ' + pronoun)

        self.pronoun_cache[author.lower()] = pronoun

        return pronoun

    def parse_string(self, input_string, author, user_message):
        replacements = {
            "<name>": self.username,
            "<channel>": self.channel[1:],
            "<game>": self.get_game(self.channel_id),
            "<author>": author,
            "<emotes>": ', '.join(map(str, self.emotes)),
            "<UTC>": str(datetime.now(timezone.utc)),
            "<time>": str(datetime.now()),
            "<chatter_pronouns>": self.get_pronouns(author),
            "<streamer_pronouns>": self.get_pronouns(self.channel[1:]),
            "<users>": ', '.join(map(str, self.get_users()))
        }

        for placeholder, replacement in replacements.items():
            input_string = input_string.replace(placeholder, replacement)

        sentences = input_string.split('. ')
        parsed_list = [{"role": "system", "content": sentence} for sentence in sentences]

        for m in self.message_queue:
            if m.split(': ')[0] == self.username.lower():
                parsed_list.append({"role": "assistant", "content": m.split(': ')[1]})
            elif m.split(': ')[1] != user_message:
                parsed_list.append({"role": "system", "name": m.split(': ')[0], "content": m.split(': ')[1]})

        if app.openai_api_model.get() == 'mpt-7b-chat' or app.openai_api_model.get() == 'WizardLM-13B':
            return parsed_list
        parsed_list.append({"role": "user", "name": author, "content": user_message})
        return parsed_list

    def send_message_delayed(self, message, delay_seconds, **kwargs):
        print('Called send_message_delayed ' + message + ' in ' + delay_seconds + ' seconds')
        app.append_to_log('Called send_message_delayed ' + message + ' in ' + delay_seconds + ' seconds')

        def delayed_print():
            seconds = int(delay_seconds)
            while seconds > 0 and app.bot_running:
                time.sleep(1)
                seconds -= 1
            if app.bot_running:
                self.connection.privmsg(self.channel, message)
                app.append_to_log(self.username + ': ' + message)
                print(self.username + ': ' + message)
                self.message_queue.append(self.username + ': ' + message)

        thread = threading.Thread(target=delayed_print)
        thread.start()

        return 'Timer Set'

    def on_disconnect(self, c, e):
        self.message_queue.clear()
        print('Disconnected')
        app.append_to_log('Disconnected')

    def on_ctcp(self, c, e):
        nick = e.source.nick
        if e.arguments[0] == "VERSION":
            c.ctcp_reply(nick, "VERSION " + self.get_version())
        elif e.arguments[0] == "PING":
            if len(e.arguments) > 1:
                c.ctcp_reply(nick, "PING " + e.arguments[1])
        elif (
                e.arguments[0] == "DCC"
                and e.arguments[1].split(" ", 1)[0] == "CHAT"
        ):
            self.on_dccchat(c, e)
        message = e.arguments[1]
        author = ''
        for tag in e.tags:
            if tag['key'] == 'display-name':
                author = tag['value']
                break
        print(author + " " + message)
        app.append_to_log((author + " " + message))

    def on_pubmsg(self, c, e):
        message = e.arguments[0]

        author = ''
        for tag in e.tags:
            if tag['key'] == 'display-name':
                author = tag['value']
                break
        print(author + ": " + message)
        app.append_to_log(author + ": " + message)
        self.message_queue.append(author + ": " + message)
        self.last_message[author.lower()] = message

        if author.lower() not in str(app.user_list.get(0, tk.END)):
            self.users.append(author.lower())
            app.user_list.insert(tk.END, author.lower())
            app.user_count.config(text=app.user_list.size())

        # If a chat message starts with an exclamation point, try to run it as a command
        if e.arguments[0].startswith('!'):
            cmd = e.arguments[0][1:].split()
            if len(cmd) > 0:
                print('Received command: ' + cmd[0])
                app.append_to_log('Received command: ' + cmd[0])
                self.do_command(e, cmd)
            return

        rand_chat = random.random()
        if app.mute and rand_chat > float(app.frequency_slider.get()) / 100:
            return

        elif message.lower() == (self.username + " yes").lower() or message.lower() == \
                ('@' + self.username + " yes").lower():
            c.privmsg(self.channel, ":)")
            app.append_to_log(self.username + ': ' + ":)")
        elif message.lower() == (self.username + " no").lower() or message.lower() == \
                ('@' + self.username + " no").lower():
            c.privmsg(self.channel, ":(")
            app.append_to_log(self.username + ': ' + ":(")
        elif message.lower().startswith(("thanks " + self.username).lower()) or \
                message.lower().startswith(("thanks @" + self.username).lower()):
            c.privmsg(self.channel, "np")
            app.append_to_log(self.username + ': ' + "np")
        else:
            if rand_chat <= float(app.frequency_slider.get()) / 100 or self.username.lower() in message.lower() or \
                    "@" + self.username.lower() in message.lower():
                thread = threading.Thread(target=lambda: self.generate_response(author, message))
                thread.start()

    def generate_response(self, author, message):
        self.input_text = app.input_text.get('1.0', 'end')

        if app.openai_api_model.get() == 'mpt-7b-chat' or app.openai_api_model.get() == 'WizardLM-13B':
            try:
                if app.openai_api_model.get() == 'mpt-7b-chat':
                    gmll_model = 'ggml-mpt-7b-chat.bin'
                else:
                    gmll_model = 'wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin'

                with io.StringIO() as buffer, redirect_stdout(buffer):
                    self.model4a = gpt4all.GPT4All(model_name=gmll_model,
                                                   model_path=os.path.abspath('.'),
                                                   allow_download=False)
                    output = buffer.getvalue().strip()
                app.append_to_log(output)
                print(output)

                with self.model4a.chat_session():
                    self.model4a.current_chat_session = self.parse_string(self.input_text, author, message)
                    response = self.model4a.generate(message, max_tokens=500, temp=0.7).encode('ascii',
                                                                                               'ignore').decode('ascii')

                response = response.strip().replace('\r', ' ').replace('\n', ' ')
                while response.startswith('.') or response.startswith('/'):
                    response = response[1:]
                if response.lower().startswith(self.username.lower()):
                    response = response[len(self.username + ': '):]
                while len(('PRIVMSG' + self.channel + " " + response + '\r\n').encode()) > 488:
                    response = response[:-1]

                self.connection.privmsg(self.channel, response[:500])
                app.append_to_log(self.username + ': ' + response[:500])
                print(self.username + ': ' + response[:500])
                self.message_queue.append(self.username + ': ' + response[:500])

            except Exception as e:
                print(str(e))
                print(traceback.format_exc())
                app.append_to_log(str(e))
                app.append_to_log(traceback.format_exc())

        else:
            retry = 0
            while retry < 3:
                message_array = self.parse_string(self.input_text, author, message)

                try:
                    response = openai.ChatCompletion.create(model=app.openai_api_model.get(),
                                                            messages=message_array,
                                                            functions=self.functions,
                                                            user=self.channel[1:]
                                                            )

                    response_message = response["choices"][0]["message"]

                    # Step 2: check if GPT wanted to call a function
                    if response_message.get("function_call"):
                        # Step 3: call the function
                        # Note: the JSON response may not always be valid; be sure to handle errors
                        available_functions = {
                            # "get_user_pronouns": self.get_pronouns,
                            "get_launch": self.get_launch,
                            "get_users": self.get_users,
                            "get_stream": self.get_stream,
                            "get_game_info": self.get_game_info,
                            "send_message_delayed": self.send_message_delayed,
                        }  # only one function in this example, but you can have multiple
                        function_name = response_message["function_call"]["name"]
                        function_to_call = available_functions[function_name]
                        function_args = json.loads(response_message["function_call"]["arguments"])
                        function_response = function_to_call(
                            # author=function_args.get("user"),
                            when=function_args.get("when"),
                            streamer=function_args.get("streamer"),
                            game=function_args.get("game"),
                            message=function_args.get("message"),
                            delay_seconds=function_args.get("delay_seconds")
                        )

                        # Step 4: send the info on the function call and function response to GPT
                        message_array.append(response_message)  # extend conversation with assistant's reply
                        # noinspection PyTypeChecker
                        message_array.append(
                            {
                                "role": "function",
                                "name": function_name,
                                "content": function_response,
                            }
                        )  # extend conversation with function response
                        response = openai.ChatCompletion.create(
                            model=app.openai_api_model.get(),
                            messages=message_array,
                        )  # get a new response from GPT where it can see the function response

                    if hasattr(response, 'choices'):
                        response.choices[0].message.content = \
                            response.choices[0].message.content.strip().replace('\r', ' ').replace('\n', ' ')
                        while response.choices[0].message.content.startswith('.') or \
                                response.choices[0].message.content.startswith('/'):
                            response.choices[0].message.content = response.choices[0].message.content[1:]
                        if response.choices[0].message.content.lower().startswith(self.username.lower()):
                            response.choices[0].message.content = response.choices[0].message.content[
                                                                  len(self.username + ': '):]
                        while len(('PRIVMSG' + self.channel + " " + response.choices[
                            0].message.content + '\r\n').encode()) > 488:
                            response.choices[0].message.content = response.choices[0].message.content[:-1]
                        self.connection.privmsg(self.channel, response.choices[0].message.content[:500])
                        app.append_to_log(self.username + ': ' + response.choices[0].message.content[:500])
                        print(self.username + ': ' + response.choices[0].message.content[:500])
                        self.message_queue.append(self.username + ': ' + response.choices[0].message.content[:500])
                        break
                    else:
                        print(response)
                        app.append_to_log(response)

                except Exception as e:
                    retry += 1
                    print(str(e))
                    print(traceback.format_exc())
                    app.append_to_log(str(e))
                    app.append_to_log(traceback.format_exc())

    def do_command(self, e, cmd):
        c = self.connection
        if len(cmd) == 2:
            if cmd[0] == self.username and cmd[1] == 'version':
                c.privmsg(self.channel, get_version() + ' ' + app.openai_api_model.get())
                app.append_to_log(self.username + ': ' + get_version() + ' ' + app.openai_api_model.get())
                print(self.username + ': ' + get_version() + ' ' + app.openai_api_model.get())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyWiki Lite")
    parser.add_argument("--version", action="store_true", help="Show the version number")
    args = parser.parse_args()

    if args.version:
        print(get_version())
        sys.exit()

    app = TwitchBotGUI()
    app.mainloop()
