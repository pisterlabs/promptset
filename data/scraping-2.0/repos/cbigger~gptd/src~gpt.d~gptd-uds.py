#!/usr/bin/env python3
import socketserver
import threading
import os
import openai
from dotenv import load_dotenv
import logging
import codecs
import traceback
from systemd import journal

logger = logging.getLogger(__name__)
logger.addHandler(journal.JournalHandler(SYSLOG_IDENTIFIER='gptd-uds'))

config_file = "/etc/gpt.d/.env"
if not os.path.exists(config_file):
    logger.error("Configuration file not found. Please update your configuration in /etc/gpt.d/gptd.conf")
    print(f'Config not found at {config_file}')
    exit()

load_dotenv(dotenv_path=config_file)

gpt_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
chat_model = os.getenv("CHAT_MODEL")

class OpenAI_Chat:
    def __init__(self, chat_model, user_input, system_prompt="Your name is KerBI, or Kernel Bound Intelligence. You are an AI living in a linux kernel."):
        self.model = chat_model
        self.user_input = user_input
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logger.info("Instantiating: OpenAI_Chat")

    def create_chat(self, ):
        logger.info("Starting chat")
        self.messages.append({"role": "user", "content": self.user_input})

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )
        ai_response=""

        for r in response:
            try:
                ai_response=(ai_response+(r["choices"][0]["delta"]["content"]))
                yield r["choices"][0]["delta"]["content"]
            except:
                traceback.print_exc()
        self.messages.append({"role": "KerBI", "content": ai_response})
        print(self.messages)
        print(ai_response)

class ChatHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # Now process chat with the chosen instance
        while True:
            data = self.request.recv(1024).decode('utf-8').strip()
            if data:
                logger.info(f"Received data from {self.client_address}")
                chat_instance = OpenAI_Chat(chat_model, data)
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
        print("Starting server...")
        try:
            while True:
                server.handle_request()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()
            server.server_close()
        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    run_server()
