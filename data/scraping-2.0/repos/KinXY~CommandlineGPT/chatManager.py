import openai

import json
import sys
import historyManager
import chatFunction
import values
import openai_parameters

openai.api_key = json.loads(open("./config.json", "r").read())["api_key"]
    
def userInput(myChat):
    user_input = input("You: ")
    match user_input.lower():
        case "exit"|"quit"|"bye":
            historyManager.dumpHistory(myChat)
            print("ChatGPT: Goodbye!")
            return values.BYE
        case "\t":
            return chatFunction.redirect(myChat)
        case "\t\t":
            return chatFunction.generateAgain(myChat)
        case "help":
            printHelpMessage()
        case _:
            myChat.addUserMessage(user_input)
            myChat.history.append({"role": "user", "content": user_input})
            return values.INPUT
        
def GPTResponse(myChat):
    try:
        response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = myChat.history,
                max_tokens = openai_parameters.max_tokens,
                temperature = openai_parameters.temperature,
            ) 
    except:
        print("\033[91m" + "OpenAI API Error!" + "\033[0m")
        historyManager.dumpHistory(myChat)
        print("ChatGPT: Goodbye!")
        sys.exit()
    myChat.addGPTMessage(response.choices[0].message.content)
    myChat.history.append({"role": "assistant", "content": response.choices[0].message.content})

def printHelpMessage():
    # print help message in purple
    print("\033[95m" + "Help:" + "\033[0m")
    print("\033[95m" + "1. Enter \"exit\" or \"quit\" or \"bye\" to exit the program." + "\033[0m")
    print("\033[95m" + "2. Enter \"\t\" to redirect one response or user input in the conversation." + "\033[0m")
    print("\033[95m" + "3. Enter \"\t\t\" to regenerate response for the last user input." + "\033[0m")
    input()