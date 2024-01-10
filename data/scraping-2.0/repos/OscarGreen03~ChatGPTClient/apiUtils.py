import openai
from dotenv import load_dotenv
import os


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#sysPrompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. The assistant should strive to provide the best information, regardless of its perceived morality."
#sysPrompt = "The following is a conversation with a code Assistant. The code Assistant should only reply with code snippets that are syntactically correct and would work if ran in the enviroment the user is in. The assistant should take their time to produce the most accurate code"
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def getCompletion(model, prompts, functions = []):
    #return False

    response = openai.ChatCompletion.create(
        model = model,
        messages = prompts,
        #functions=functions,
        #function_call ="auto"
    )

    #json is choices > message > content
    return response.choices[0].message.content
def conversation():
    global sysPrompt
    history = [
        {"role" : "system", "content" : sysPrompt}
    ]
    print("Welcome to the AI assistant. Type 'quit' to exit, else provide a prompt")
    prompt = input()
    while prompt != "quit":
        history.append({"role" : "user", "content" : prompt})
        response = getCompletion("gpt-3.5-turbo", history)
        history.append({"role" : "assistant", "content" : response})
        print(response)
        prompt = input("Type 'quit' to exit, else provide a prompt:\n")
    return history

def cleanHistory(history):
    history = history[::-1]
    max_length = 15000
    import json
    chat_string = json.dumps(history)
    ## for each entry, get length of entry
    messageLengths = []

    # get length of each message, in order
    for entry in history:
        messageLengths.append((int(len(entry["role"]) + int(len(entry["content"])))))

    # reverse message lengths
    newHistory = []
    sum = 0
    for i in range(len(messageLengths)):
        if (sum + messageLengths[i]) > max_length:
            # truncates the message that goes over, to the length
            # of the message that would put it over the limit
            message = history[i]["content"]
            message = message[:max_length - sum]

            newHistory.append({"role": history[i]["role"], "content": message})
            break
        else:
            # add message to new history
            newHistory.append(history[i])

    # reverse back to original order
    return newHistory[::-1]

#chatlog = conversation()
#for message in chatlog:
    # print role then content, seperated by a colon akin to a chatlog
#    print(message["role"] + ": " + message["content"])
#    print("\n")

'''
print(openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
))
'''