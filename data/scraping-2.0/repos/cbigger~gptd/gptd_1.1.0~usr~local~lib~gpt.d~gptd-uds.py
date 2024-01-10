#!/usr/bin/env python3
import socketserver
import threading
import os
#from simpleaichat import simpleaichat  # import AIChat
import openai
from dotenv import load_dotenv
import logging
import codecs
import traceback


logging.basicConfig(filename='uds_server.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def logout(string):
    print(string)
    logging.info(string)

config_file = "/etc/gpt.d/.env"
if not os.path.exists(config_file):
    logout("Configuration file not found. Please update your configuration in /etc/gpt.d/gptd.conf")
    exit()

load_dotenv(dotenv_path=config_file)

gpt_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
chat_model = os.getenv("CHAT_MODEL")
ai_text_colour = codecs.decode(os.getenv("AI_TEXT_COLOUR"), 'unicode_escape')
ai_name_colour = codecs.decode(os.getenv("AI_NAME_COLOUR"), 'unicode_escape')
text_reset = codecs.decode(os.getenv("TEXT_RESET"), 'unicode_escape')


class OpenAI_Chat:
    def __init__(self, model, user_input, system_prompt="You are a Kernel Integrated Artificial Intelligence, an AI running in a linux kernel."):
        self.model = model
        self.user_input = user_input
        self.system_prompt = system_prompt

    def create_chat(self):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_input}
        ]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        for r in response:
            try:
                yield r["choices"][0]["delta"]["content"]
            except Exception as e:
                traceback.print_exc()


class ChatHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # setup default, which is shared rn
        data = self.request.recv(1024).decode('utf-8').strip()


        # Now process chat with the chosen instance
        while True:
            data = self.request.recv(1024).decode('utf-8').strip()
            if data:
                logout(f"Received from {self.client_address}: {data}")
                last_answer = ""
                answer = ""
                chat_instance = OpenAI_Chat(chat_model, data)

               # response = (text_reset + "\n")
                print(data)
                for chunk in chat_instance.create_chat():
                    self.request.sendall(chunk.encode('utf-8'))
            else:
                break

class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

def run_server():
    socket_file_path = "/tmp/gptd.sock"
    if os.path.exists(socket_file_path):
        os.remove(socket_file_path)
    with ThreadedUnixServer(socket_file_path, ChatHandler) as server:
        server.timeout = 1
        logout("Starting server...")
        try:
            while True:
                server.handle_request()
        except KeyboardInterrupt:
            logout("\nShutting down server...")
            server.shutdown()
            server.server_close()
        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    run_server()
