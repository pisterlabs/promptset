# Implement a function to interact with the OpenAI API.
import openai
import os
from copy import deepcopy as dcopy
from .conf_openai import *

# import sleep
from time import sleep

def set_proxy():
    os.environ['HTTP_PROXY'] = "http://10.30.69.253:7890"
    os.environ['HTTPS_PROXY'] = "http://10.30.69.253:7890"
    print("Proxy set.")

openai.api_key=key
openai.proxy=proxies
# # get runtime variable from command line
# import sys
# args = sys.argv
# try: 
#     testing = ("--debug" in args or "-D" in args)
# except:
#     testing = False
# # get tempreture from command line
# try:
#     temperature = float(args[args.index("-T")+1])
# except:
#     temperature = 0.7
# try: 
#     gpt4 = ("--gpt4" in args or "-G4" in args)
# except:
#     gpt4 = False

# print confguration
print("Testing:",testing)
print("Temperature:",temperature)
print("GPT-4:",gpt4)

# # another version to handle server errors
def chat_user_input(prompt,messages,chatlog,temperature=0.7,retry=False):
    messages.append({"role": "user", "content": prompt})
    model = "gpt-4" if gpt4 else "gpt-3.5-turbo"
    if retry:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            restext = response.choices[0]["message"]
            messages.append({"role": "assistant", "content": restext["content"]})
            chatlog.append(dcopy(messages))
            return restext["content"]
        except Exception as e:
            print("OpenAI API error, please retry.")
            # print error message
            print(e)
            return None
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
        )
        restext = response.choices[0]["message"]
        messages.append({"role": "assistant", "content": restext["content"]})
        chatlog.append(dcopy(messages))
        return restext["content"]

def chat_sys_function(sys_input,messages,chatlog):
    messages.append({"role": "system", "content": sys_input})
    chatlog.append(dcopy(messages))
    pass 
    

def chat_msg_printer(messages):
    print("Messages:")
    if testing:
        print(messages)
    for i in range(len(messages)):
        print("(",format(i,"2d"),"-",sep="",end="")
        if messages[i]["role"] == "user":
            print("U):\t",end="")
        else:
            print("A):\t",end="")
        print(messages[i]["content"])

def chat_log_printer(logs):
    print("Logs:")
    if testing:
        print(logs)
    for i in range(len(logs)):
        print("Log[",i,"]:",sep="")
        print('-'*len(logs[i]),end="")
        chat_msg_printer(logs[i])
    print("End of logs.")

CAHT_RESERVED_STRINGS = {
    "?": "Show this message.",
    "t": '\"Reply with 2 random numbers like $1,2$.\"',
    ">>": "procced",
    "<<": "delete the input",
    "<<<": "delete the last message",
    "SL": "show the logs and choose one",
    "LS": "list the logs",
}

if __name__ == "__main__":
    messages = []
    logs = []
    while True:
        try:
            print("User:")
            inputString = ""
            inputS = input()
            
            # Handle Controller
            while (inputS not in CAHT_RESERVED_STRINGS.keys()):
                if testing:
                    print(inputS)
                inputString = inputString + "\n" +inputS
                inputS = input()
            
            # Handle Reserved Strings
            if inputS == ">>":
                pass
            elif inputS == "<<":
                chat_msg_printer(messages)
                continue
            elif inputS == "<<<":
                messages.pop()
                messages.pop()
                logs.append(messages)
                continue
            elif inputS == "SL":
                if len(logs) == 0:
                    print("No logs.")
                    continue
                chat_log_printer(logs)
                print("Choose one log to continue:")
                while True:
                    try:
                        log_index = int(input())
                        messages = dcopy(logs[log_index])
                        break
                    except (ValueError, IndexError):
                        print("Bad input, please retry.")
                continue
            elif inputS == "LS":
                if len(logs) == 0:
                    print("No logs.")
                    continue
                chat_log_printer(logs)
                continue
            elif inputS == "?":
                print("Reserved Strings:")
                for i in CAHT_RESERVED_STRINGS.keys():
                    print(i,":",CAHT_RESERVED_STRINGS[i])
                continue
            elif inputS == "t":
                inputString = "Reply with 2 random numbers like $1,2$."

        except (KeyboardInterrupt, EOFError):
            print("\n\nBye!\n")
            exit(0)
        if testing:
            print("inputString:",inputString)
        print("Waiting>>",end="",flush=True)
        # sleep(0.5)
        # display the realtime waiting animation
        chat_user_input(
            inputString,
            messages,
            logs,
        )
        print("Assistant: ",messages[-1]["content"],"\n")


# # user_input(
# #     inputString,
# #     messages,
# #     logs,
# # ) 

# # print(messages[-1]["content"])
