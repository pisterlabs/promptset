#!/usr/bin/env python3
import math
import subprocess
import sys
import threading
import time
import traceback
from enum import Enum
from io import StringIO

import openai
import os

import tiktoken as tiktoken

try:
    openai.api_key = os.environ['openai_api_key']
except KeyError:
    try:
        openai.api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        print("OpenAI API Key not set as environment variable named: \"openai_api_key\"! Visit "
              "https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety for more info.")
        exit()
    print("OpenAI API Key not set as environment variable named: \"openai_api_key\"! Visit "
          "https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety for more info.")
    exit()

max_num_tokens = 3000


class Role(Enum):
    SYSTEM = "system"
    USER = "user"


max_idle_time = 500
idle = 0

# gpt_sleep_time = 10
gpt_sleep_time_sem = threading.Semaphore(1)
pause = False
pause_sem = threading.Semaphore(1)

# show_server_output = False
show_server_output_sem = threading.Semaphore(1)

messages = []
messages_sem = threading.Semaphore(1)

# configuration
try:
    with open("config.txt", "r") as f:
        try:
            for line in f:
                c = line.split(" ")
                if c[0] == "show_server_output":
                    show_server_output = int(c[1])
                if c[0] == "gpt_sleep_time":
                    gpt_sleep_time = float(c[1])
                if c[0] == "max_tokens_per_response":
                    max_tokens_per_response = int(min(9223372036854775807.0, float(c[1])))
                if c[0] == "temperature":
                    temperature = float(c[1])
                if c[0] == "token_limit":
                    token_limit = float(c[1])
            f.close()
        except ValueError:
            print("Couldn't read value from config.txt! Check that you are using the correct types.")
            f.close()
            traceback.print_exc()
            exit()
except FileNotFoundError:
    print("config.txt not found in directory! Make sure this file is stored in the same directory as this "
          "script.")
    traceback.print_exc()
    exit()
# temperature = 0.8
temperature_sem = threading.Semaphore(1)
total_tokens_sent = 0
# token_limit = math.inf
token_limit_sem = threading.Semaphore(1)
# max_tokens_per_response = math.inf
max_tokens_per_response_sem = threading.Semaphore(1)

new_output = ""
new_output_sem = threading.Semaphore(1)
lines = 0

argc = len(sys.argv)
if argc > 1:  # TODO error catching
    server_dir = sys.argv[1]
    if argc > 2:
        server_jar = sys.argv[2]
    else:
        server_jar = 'server.jar'
else:
    server_dir = '../server'
    server_jar = 'server.jar'
executable = f'java -Xmx1024M -Xms1024M -jar {server_jar}'

try:
    with open("initial_prompt.txt", "r") as f:
        initial_training_prompt = f.read()
        f.close()
except FileNotFoundError:
    print("initial_prompt.txt not found in directory! Make sure this file is stored in the same directory as this "
          "script.")
    traceback.print_exc()
    exit()

print("initial prompt: ")
print(initial_training_prompt)


def send_system_prompt(prompt):
    return send_user_prompt(prompt, Role.SYSTEM)


def send_server_update(server_output):
    return send_user_prompt(server_output)


def num_tokens_from_messages(ms, model="gpt-3.5-turbo-0301"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in ms:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


initial_num_tokens = num_tokens_from_messages([{"role": Role.SYSTEM.value, "content": initial_training_prompt}])
print("initial prompt num of tokens: " + str(initial_num_tokens))


def print_messages():
    for m in messages:
        print(m["content"])


def send_user_prompt(prompt, role=Role.USER):
    msg_obj = {"role": role.value, "content": prompt}

    # TODO handle single message with token length > max_num_tokens
    # if len(messages) > 0 and num_tokens_from_messages([msg_obj]) > max_num_tokens - initial_num_tokens:
    #     prompt = prompt[-(max_num_tokens - initial_num_tokens)]
    # msg_obj = {"role": role.value, "content": prompt}

    messages.append(msg_obj)

    while len(messages) > 1 and num_tokens_from_messages(messages) > max_num_tokens:
        print("removing oldest message from request: " + str(num_tokens_from_messages(messages)) + "/" + str(
            max_num_tokens))
        messages.pop(1)

    num_tokens = num_tokens_from_messages(messages)

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=get_messages(),
            temperature=get_temperature(),
            # max_tokens=get_max_tokens_per_response() - num_tokens,  # TODO fix
        )
    except openai.error.InvalidRequestError:
        traceback.print_exc()
        return "ERROR"

    global total_tokens_sent
    total_tokens_sent += completion["usage"]["prompt_tokens"]
    print(f"{total_tokens_sent} total tokens sent")

    if total_tokens_sent > get_token_limit():
        print("TOKEN LIMIT EXCEEDED! Pausing API calls...")
        set_pause(True)

    new_message = completion.choices[0].message
    messages.append(new_message)

    return new_message.content


def execute_server_command(command):
    process.stdin.write(command + "\n")
    process.stdin.flush()


def handle_response(r):
    print("\n\ngpt response: \n" + r + "\n\n")
    for line in StringIO(r):
        if line[0] == "":
            continue
        # elif line[0] == "/":
        #     execute_server_command(line)
        # else:
        #     execute_server_command("/say " + line)
        execute_server_command(line)


def restart_gpt():
    messages.clear()  # TODO is this atomic? might need to use semaphore
    messages.append({"role": Role.SYSTEM.value, "content": initial_training_prompt})


def input_thread():  # function to read user input and write it to the process input stream
    while True:
        user_input = input()
        if len(user_input) <= 0:
            continue
        elif user_input[0] == "!":
            handle_response(send_system_prompt(user_input))
        elif user_input[0] == ":":
            c = user_input[1:len(user_input)].split(" ")
            if c[0] == "pause":
                set_pause(True)
                print("paused gpt")
            elif c[0] == "resume":
                set_pause(False)
                print("resumed gpt")
            elif c[0] == "set":
                if c[1] == "gpt_sleep_time":
                    if len(c) < 3:
                        print("command needs second argument of type float")
                        return
                    try:
                        val = float(c[2])
                    except ValueError:
                        print("second argument should be float")
                        return
                    set_gpt_sleep_time(val)
                    print("set gpt_sleep_time to " + str(val))
                elif c[1] == "show_server_output":
                    if len(c) < 3:
                        print("command needs second argument of type integer")
                        return
                    try:
                        val = int(c[2])
                    except ValueError:
                        print("second argument should be integer")
                        return
                    new = True if val > 0 else False
                    set_show_server_output(new)
                    print("set show_server_output to " + str(new))
                elif c[1] == "max_tokens_per_response":
                    if len(c) < 3:
                        print("command needs second argument of type integer")
                        return
                    try:
                        val = float(c[2])
                    except ValueError:
                        print("second argument should be int")
                        return
                    set_max_tokens_per_response(val)
                    print("set max_tokens_per_response to " + str(val))
                elif c[1] == "temperature":
                    if len(c) < 3:
                        print("command needs second argument of type float [0.0-1.0]")
                        return
                    try:
                        val = max(2.0, min(0.0, float(c[2])))
                    except ValueError:
                        print("third argument should be float [0.0-2.0]")
                        return
                    set_temperature(val)
                    print("set temperature to " + str(val))
                elif c[1] == "token_limit":
                    if len(c) < 3:
                        print("command needs second argument of type integer")
                        return
                    try:
                        val = int(c[2])
                    except ValueError:
                        print("third argument should be an integer")
                        return
                    set_token_limit(val)
                    print("set token_limit to " + str(val))
                else:
                    print("unknown command: " + user_input)

            elif c[0] == "get":
                if c[1] == "gpt_sleep_time":
                    print("gpt_sleep_time: " + str(get_gpt_sleep_time()))
                elif c[1] == "total_tokens_sent":
                    print("total_tokens_sent: " + str(total_tokens_sent))
                elif c[1] == "temperature":
                    print("temperature: " + str(get_temperature()))
                elif c[1] == "token_limit":
                    print("token_limit: " + str(get_token_limit()))
                else:
                    print("unknown command: " + user_input)
            elif c[0] == "restart":
                restart_gpt()
            else:
                print("unknown command: " + user_input)
        else:
            execute_server_command(user_input)


def output_thread():
    while True:
        o = process.stdout.readline()
        if not o:
            print("output ended!")
            break
        for ln in StringIO(o):  # only add most relevant output to "new_output" so that tokens aren't wasted
            if get_show_server_output():
                print(ln, end="")

            # This prevents ChatGPT from responding to malformed commands over and over.
            # TODO manually enable/disable in config
            if "[Server]" in ln:
                continue
            elif ": Unknown" in ln:  # excluding this will prevent a never ending loop of GPT sending bad commands
                continue
            elif ": Incorrect arg" in ln:
                continue
            elif "<--[HERE]" in ln:
                continue

            append_new_output(ln)


def get_messages():
    messages_sem.acquire()
    val = messages
    messages_sem.release()
    return val


def set_messages(val):
    messages_sem.acquire()
    global messages
    messages = val
    messages_sem.release()


def get_show_server_output():
    show_server_output_sem.acquire()
    val = show_server_output
    show_server_output_sem.release()
    return val


def set_show_server_output(val):
    show_server_output_sem.acquire()
    global show_server_output
    show_server_output = val
    show_server_output_sem.release()


def get_pause():
    pause_sem.acquire()
    val = pause
    pause_sem.release()
    return val


def set_pause(val):
    pause_sem.acquire()
    global pause
    pause = val
    pause_sem.release()


def get_gpt_sleep_time():
    gpt_sleep_time_sem.acquire()
    val = gpt_sleep_time
    gpt_sleep_time_sem.release()
    return val


def set_gpt_sleep_time(val):
    gpt_sleep_time_sem.acquire()
    global gpt_sleep_time
    gpt_sleep_time = val
    gpt_sleep_time_sem.release()


def get_temperature():
    temperature_sem.acquire()
    val = temperature
    temperature_sem.release()
    return val


def set_temperature(val):
    temperature_sem.acquire()
    global temperature
    temperature = val
    temperature_sem.release()


def get_token_limit():
    token_limit_sem.acquire()
    val = token_limit
    token_limit_sem.release()
    return val


def set_token_limit(val):
    token_limit_sem.acquire()
    global token_limit
    token_limit = val
    token_limit_sem.release()


def get_max_tokens_per_response():
    max_tokens_per_response_sem.acquire()
    val = max_tokens_per_response
    max_tokens_per_response_sem.release()
    return val


def set_max_tokens_per_response(val):
    max_tokens_per_response_sem.acquire()
    global max_tokens_per_response
    max_tokens_per_response = val
    max_tokens_per_response_sem.release()


def get_new_output():
    return new_output


def get_and_clear_new_output():
    new_output_sem.acquire()
    global new_output
    o = new_output
    new_output = ""
    new_output_sem.release()
    return o


def set_new_output(output):
    new_output_sem.acquire()
    global new_output
    new_output = output
    new_output_sem.release()


def append_new_output(output):
    new_output_sem.acquire()
    global new_output
    new_output += output
    new_output_sem.release()


os.chdir(server_dir)
process = subprocess.Popen(executable, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

# start a separate thread to handle user input
input_thread = threading.Thread(target=input_thread)
input_thread.start()

# start a separate thread to handle server output
output_thread = threading.Thread(target=output_thread)
output_thread.start()

print("initial gpt response: \n\n" + send_system_prompt(initial_training_prompt))

while True:
    time.sleep(get_gpt_sleep_time())

    if get_pause():
        continue

    if len(get_new_output()) <= 0:
        idle += 1
        if idle * 5 > max_idle_time:
            idle = 0
            print("Server has been idle for too long! Time for some fun...")
            send_system_prompt("punish some players")
        continue
    response = send_server_update(get_and_clear_new_output())
    handle_response(response)

# TODO gracefully exit

process.wait()

input_thread.join()
output_thread.join()

process.stdin.close()
process.kill()
