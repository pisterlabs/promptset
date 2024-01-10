import threading
import time
import logging
import sys
import datetime

from rutines import RutineScheduler
from extensions import get_extensions
from openai import OpenAI
from run import Run
from extensions.jess_extension import jess_extension, get_openai_spec


logging.basicConfig(level=logging.ERROR, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_jeff_instructions():
    with open("jess_prompt.txt") as f:
        return f.read()
    return ""


instructions = read_jeff_instructions()


class Jess(object):

    def __init__(self, client, assistent, message_handler, jess_chat_thread, extensions) -> None:
        self.run = None
        self.thread = jess_chat_thread
        self.assistent = assistent
        self.stopped = False
        self.client = client
        self.next_action_time = None
        self.next_message = None
        self.message_handler = message_handler
        self.extensions = extensions

    def stop(self):
        self.stopped = True

    def _cancel_scheduled_message(self):
        self.next_action_time = None
        self.next_message = None

    def send_message(self, message_to_send):
        self._cancel_scheduled_message()
        self._send_message(message_to_send)
        self._send_system_message_about_action()
        
    def _send_message(self, message_to_send):
        while self.run and not self.run.finished():
            time.sleep(1)
        
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=message_to_send
        )
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistent.id
        )
        self.run = Run(run.id, self.thread.id, self.client, self, self.extensions, logger, self._on_system_message)  
        self.run.execute()      

    @jess_extension(
        description="Schedule next message to be sent to a user if there is no response from the user after a defined period of time.",
        param_descriptions={
            "sec_delay": "Amount of seconds from now when the message should be sent.",
            "message": "Message to be sent to the user."
        }
    )
    def schedule_message(self, sec_delay: int, message: str):
        self.next_action_time = time.time() + int(sec_delay)
        self.next_message = message
        return "DONE"
    
    def _on_system_message(self, message):
        self.message_handler(f"[SYSTEM]: {message}")

    def on_messages(self, messages):
        for message in messages:
            if message:
                self.message_handler(message)

    def _send_system_message_about_action(self):
        self._send_message("*SYSTEM* This is not real user message, and user will not read it. This message allows you to make schedule a pro-active message to a user, if user will not respond to you any time soon by using schedule_message action. Message that you might schedule will be canceled if the user will answer to you first. Do NOT reply to this message anything.")

    def get_scheduled_time(self):
        readable_time = datetime.datetime.fromtimestamp(self.next_action_time)

        # Format the datetime object into a readable string
        # For example, "2023-11-24 15:30:00"
        return readable_time.strftime("%Y-%m-%d %H:%M:%S")
    
    def get_scheduled_message(self):
        return self.next_message

    def execute(self):
        logger.debug("Starting Jess")
        while not self.stopped:
            logger.debug("Checking run status")
            if self.next_action_time and time.time() >= self.next_action_time:
                logger.debug("now")
                self.on_messages([self.next_message])
                self.next_action_time = None
                self.next_message = None
                self._send_system_message_about_action()
            time.sleep(2)

    def drop_chat_thread(self):
        self._cancel_scheduled_message()
        self.thread = self.client.beta.threads.create(
            messages=[]
        )        

    @staticmethod
    def start(message_handler, extensions):
        JESS_NAME = "Jess"
        client = OpenAI()
        tools = [get_openai_spec(Jess.schedule_message)]
        for extension_name in extensions.keys():
            tools.append(get_openai_spec(extensions[extension_name]))
        jess_assitent_args = {
            "name": JESS_NAME,
            "instructions": instructions,
            "tools": tools,
            "model": "gpt-4-1106-preview"
        }
        jess_assitent = None
        for assistant in client.beta.assistants.list():
            if assistant.name == JESS_NAME:
                jess_assitent = assistant
                logger.debug("Found assistant")
                break

        jess_chat_thread = client.beta.threads.create(
            messages=[]
        )

        if not jess_assitent:
            logger.debug("Creating new assistant")
            jess_assitent = client.beta.assistants.create(
                **jess_assitent_args
            )
        else:
            logger.debug("Updating assistant")
            jess_assitent = client.beta.assistants.update(
                assistant_id=jess_assitent.id,
                **jess_assitent_args
            )
        jess = Jess(client, jess_assitent, message_handler, jess_chat_thread, extensions)
        jess.rutine = RutineScheduler.start_rutines(jess)
        extensions["schedule_message"] = jess.schedule_message
        thread = threading.Thread(target=jess.execute)
        thread.start()
        return jess


def message_handler(message):
    print("jess: " + message + "\n")


if __name__ == "__main__":
    extensions = get_extensions()
    jess = Jess.start(message_handler, extensions)
    # while loop with the read from keyboard and send message
    while True:
        message_to_send = input("You: ")
        jess.send_message(message_to_send)
        time.sleep(3)