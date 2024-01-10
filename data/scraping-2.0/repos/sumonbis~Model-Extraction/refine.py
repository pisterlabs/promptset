import sys
import json
import openai
import signal
import os
import traceback
import time
import traceback


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError()

class ChatGPTCall(object):
    def __init__(self, api_key_file="api_key.txt", model_name="gpt-3.5-turbo"):
        self.api_key = self.load_api(api_key_file)
        self.model_name = model_name

    @staticmethod
    def load_api(api_key_file):
        if not os.path.exists(api_key_file):
            raise ValueError(
                f"API key not found: {api_key_file}.")
        with open(api_key_file, "r") as file:
            api_key = file.read().split("\n")[0]
        return api_key

    def ask_gpt(self, query):
        openai.api_key = self.api_key
        try:
            res = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    # {"role": "user", "content": "Previous Prompts"},
                    # {"role": "user", "content": "Previous Queries"},
                    {"role": "system", "content": "You are a helpful assistant that understands C++ code."},
                    {"role": "user", "content": f"{query}"},
                ],
                # temperature=0
            )
            res = res.choices[0].message.content
        except TimeoutError:
            signal.alarm(0)  # reset alarm
            return "TIMEOUT, No response from GPT for 5 minutes"
        except Exception:
            print(traceback.format_exc())
            print("Error during querying, sleep for one minute")
            time.sleep(60)  # sleep for 1 minute
            res = self.ask_gpt(query)
        return res

    def query(self, query, timeout=60*5):
        """
        The query function for user to send query to the chatGPT
        :param query: the user's prompt
        :param timeout: the timeout for the user's query
        :return: the GPT's response
        """
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # set timeout to 5 minutes
        try:
            response = self.ask_gpt(query)
        except TimeoutError:
            print("TIMEOUT ERROR!!!")
            return "TIMEOUT, No Response From GPT For 5 minutes"
        finally:
            signal.alarm(0)  # reset alarm
        return response


def test_timeout():
    chatGPTCall = ChatGPTCall()
    res = chatGPTCall.query("noop", timeout=1)
    assert res is None
    print("Test Timeout Pass!")

def get_answer(q):
    chatGPTCall = ChatGPTCall()
    res = chatGPTCall.query(q)
    # assert res is None
    return res


def test_api_key():
    try:
        chatGPTCall = ChatGPTCall("fake_path")
    except:
        print("Test Invalid API Key Pass")
    else:
        raise ValueError("Test Fail For Invalid API Key!")

def load_query(path):
        if not os.path.exists(path):
            raise ValueError(
                f"The API Key File Is Not Found: {api_key_file}. Please Create It And Store Your API Key Here.")
        with open(path, "r") as file:
            query = file.read()
        return query

def write_to_file(file_path, text):
    try:
        with open(file_path, "w") as file:
            file.write(text)
        print("Saved to file.")
    except IOError:
        print("An error occurred while writing to the file.")

if __name__ == "__main__":
    arguments = sys.argv

    if len(arguments) > 1:
        aadl_code = arguments[1]
    else:
        print("No argument provided.")

    arch_cmd = 'Refine the AADL code and add two components that are the Ground Station, and the Drone.\n'

    code = load_query(aadl_code)
    print('Generating aadl ...')
    arch_desc = get_answer(arch_cmd + code)
    write_to_file("refined-aadl.txt", arch_desc)