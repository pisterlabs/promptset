# adapted from https://github.com/rawandahmad698/PyChatGPT

# Builtins
import sys
import time
import os
from queue import Queue
import json
from typing import Tuple

# Local
# from pychatgpt.classes import openai as OpenAI
from pychatgpt.classes import chat as ChatHandler
from pychatgpt.classes import exceptions as Exceptions
from pychatgpt.main import Options
from pychatgpt.main import Chat
from .auth import Auth

# Fancy stuff
from colorama import Fore


def token_expired(email_address) -> bool:
    f"""
        Check if the creds have expired for {email_address}
        returns:
            bool: True if expired, False if not
    """
    try:
        # Get path using os, it's in ./classes/auth.json
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, f"{email_address}_auth.json")

        with open(path, 'r') as f:
            creds = json.load(f)
        expires_at = float(creds['expires_at'])
        if time.time() > expires_at + 3600:
            return True
        else:
            return False
    except KeyError:
        return True
    except FileNotFoundError:
        return True


def get_access_token(email_address) -> Tuple[str or None, str or None]:
    """
        Get the access token for {email_address}
        returns:
            str: The access token
    """
    try:
        # Get path using os, it's in ./Classes/auth.json
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, f"{email_address}_auth.json")

        with open(path, 'r') as f:
            creds = json.load(f)
            return creds['access_token'], creds['expires_at']
    except FileNotFoundError:
        return None, None


class FastChat(Chat):
    def __init__(self,
                 email: str,
                 password: str,
                 options: Options or None = None,
                 conversation_id: str or None = None,
                 previous_convo_id: str or None = None):
        self.email = email
        self.password = password
        self.options = options

        self.conversation_id = conversation_id
        self.previous_convo_id = previous_convo_id

        self.auth_access_token: str or None = None
        self.auth_access_token_expiry: int or None = None
        self.__chat_history: list or None = None

        self._setup()

    @staticmethod
    def _create_if_not_exists(file: str):
        if not os.path.exists(file):
            with open(file, 'w') as f:
                f.write("")

    def _setup(self):
        if self.options is not None:
            # If track is enabled, create the chat log and id log files if they don't exist
            if not isinstance(self.options.track, bool):
                raise Exceptions.PyChatGPTException("Options to track conversation must be a boolean.")

            if self.options.track:
                if self.options.chat_log is not None:
                    self._create_if_not_exists(self.options.chat_log)
                    self.options.id_log = os.path.abspath(self.options.chat_log)
                else:
                    # Create a chat log file called chat_log.txt
                    self.options.chat_log = "chat_log.txt"
                    self._create_if_not_exists(self.options.chat_log)

                if self.options.id_log is not None:
                    self._create_if_not_exists(self.options.id_log)
                    self.options.id_log = os.path.abspath(self.options.id_log)
                else:
                    # Create a chat log file called id_log.txt
                    self.options.id_log = "id_log.txt"
                    self._create_if_not_exists(self.options.id_log)

            if self.options.proxies is not None:
                if not isinstance(self.options.proxies, dict):
                    if not isinstance(self.options.proxies, str):
                        raise Exceptions.PyChatGPTException("Proxies must be a string or dictionary.")
                    else:
                        self.proxies = {"http": self.options.proxies, "https": self.options.proxies}
                        print(f"{Fore.GREEN}>> Using proxies: True.")

            if self.options.track:
                print(f"{Fore.GREEN}>> Tracking conversation enabled.")
                if not isinstance(self.options.chat_log, str) or not isinstance(self.options.id_log, str):
                    raise Exceptions.PyChatGPTException(
                        "When saving a chat, file paths for chat_log and id_log must be strings.")
                elif len(self.options.chat_log) == 0 or len(self.options.id_log) == 0:
                    raise Exceptions.PyChatGPTException(
                        "When saving a chat, file paths for chat_log and id_log cannot be empty.")

                self.__chat_history = []
        else:
            self.options = Options()

        if not self.email or not self.password:
            print(f"{Fore.RED}>> You must provide an email and password when initializing the class.")
            raise Exceptions.PyChatGPTException("You must provide an email and password when initializing the class.")

        if not isinstance(self.email, str) or not isinstance(self.password, str):
            print(f"{Fore.RED}>> Email and password must be strings.")
            raise Exceptions.PyChatGPTException("Email and password must be strings.")

        if len(self.email) == 0 or len(self.password) == 0:
            print(f"{Fore.RED}>> Email cannot be empty.")
            raise Exceptions.PyChatGPTException("Email cannot be empty.")

        if self.options is not None and self.options.track:
            try:
                with open(self.options.id_log, "r") as f:
                    # Check if there's any data in the file
                    if len(f.read()) > 0:
                        self.previous_convo_id = f.readline().strip()
                        self.conversation_id = f.readline().strip()
                    else:
                        self.conversation_id = None

            except IOError:
                raise Exceptions.PyChatGPTException("When resuming a chat, conversation id and previous conversation id in id_log must be separated by newlines.")
            except Exception:
                raise Exceptions.PyChatGPTException("When resuming a chat, there was an issue reading id_log, make sure that it is formatted correctly.")

        # Check for access_token & access_token_expiry in env
        if token_expired(self.email):
            print(f"{Fore.RED}>> Access Token missing or expired for {self.email}."
                  f" {Fore.GREEN}Attempting to create them...")
            self._create_access_token()
        access_token, expiry = get_access_token(self.email)
        self.auth_access_token = access_token
        self.auth_access_token_expiry = expiry

        try:
            self.auth_access_token_expiry = int(self.auth_access_token_expiry)
        except ValueError:
            print(f"{Fore.RED}>> Expiry is not an integer.")
            raise Exceptions.PyChatGPTException("Expiry is not an integer.")

        if self.auth_access_token_expiry < time.time():
            print(f"{Fore.RED}>> Your access token is expired for {self.email}. {Fore.GREEN}Attempting to recreate it...")
            self._create_access_token()

    def _create_access_token(self) -> bool:
        openai_auth = Auth(email_address=self.email, password=self.password, proxy=self.options.proxies)
        openai_auth.create_token()

        # If after creating the token, it's still expired, then something went wrong.
        is_still_expired = token_expired(self.email)
        if is_still_expired:
            print(f"{Fore.RED}>> Failed to create access token for {self.email}.")
            return False

        # If created, then return True
        return True

    def ask(self, prompt: str, rep_queue: Queue or None = None) -> str or None:
        if prompt is None:
            print(f"{Fore.RED}>> Enter a prompt.")
            raise Exceptions.PyChatGPTException("Enter a prompt.")

        if not isinstance(prompt, str):
            raise Exceptions.PyChatGPTException("Prompt must be a string.")

        if len(prompt) == 0:
            raise Exceptions.PyChatGPTException("Prompt cannot be empty.")

        if rep_queue is not None and not isinstance(rep_queue, Queue):
            raise Exceptions.PyChatGPTException("Cannot enter a non-queue object as the response queue for threads.")

        # Check if the access token is expired
        if token_expired(self.email):
            print(f"{Fore.RED}>> Your access token for {self.email} is expired. {Fore.GREEN}Attempting to recreate it...")
            did_create = self._create_access_token()
            if did_create:
                print(f"{Fore.GREEN}>> Successfully recreated access token.")
            else:
                print(f"{Fore.RED}>> Failed to recreate access token for {self.email}.")
                raise Exceptions.PyChatGPTException(f"Failed to recreate access token for {self.email}.")

        # Get access token
        access_token = get_access_token(self.email)

        answer, previous_convo, convo_id = ChatHandler.ask(auth_token=access_token,
                                                           prompt=prompt,
                                                           conversation_id=self.conversation_id,
                                                           previous_convo_id=self.previous_convo_id,
                                                           proxies=self.options.proxies)

        if rep_queue is not None:
            rep_queue.put((prompt, answer))

        if answer == "400" or answer == "401":
            print(f"{Fore.RED}>> Failed to get a response from the API with {self.email}.")
            return None

        self.conversation_id = convo_id
        self.previous_convo_id = previous_convo

        if self.options.track:
            self.__chat_history.append("You: " + prompt)
            self.__chat_history.append("Chat GPT: " + answer)
            self.save_data()

        return answer

    def save_data(self):
        if self.options.track:
            try:
                with open(self.options.chat_log, "a") as f:
                    f.write("\n".join(self.__chat_history) + "\n")

                with open(self.options.id_log, "w") as f:
                    f.write(str(self.previous_convo_id) + "\n")
                    f.write(str(self.conversation_id) + "\n")

            except Exception as ex:
                print(f"{Fore.RED}>> Failed to save chat and ids to chat log and id_log."
                      f"{ex}")
            finally:
                self.__chat_history = []
