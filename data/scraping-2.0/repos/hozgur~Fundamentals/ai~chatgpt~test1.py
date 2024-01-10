import openai
import os
import re
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
print("Welcome to the OpenAI assistant. Type exit to exit.")
print(os.getenv("OPENAI_KEY"))
history = []

make_shorter = " make your response shorter no description only answer."
make_shorter_tr = " cevabınızı kısaltın açıklama yok sadece cevap."

def get_code3(text):
    pattern1 = r'```(?:python)?\n(.*?)```'
    pattern2 = r'```(?:python)?(.*?)```'

    # Find code snippets using both patterns
    snippets1 = re.findall(pattern1, text, re.DOTALL)
    snippets2 = re.findall(pattern2, text, re.DOTALL)

    # Combine the results and return them
    result = snippets1 + snippets2
    return result[0] if result else ""

def get_code2(text):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":text},{"role":"user","content":"get python code from last message. write only code. if there is no python code say only no code."}])
    reply = response["choices"][0].message.content
    if "no code" in reply.lower():
        return ""
    return reply

def get_code(text):
    code = get_code3(text)
    if not code:
        code = get_code2(text)
        code = get_code3(code)
    return code

# function that counts characters in history array
def count():
    count = 0
    for i in history:
        count += len(i["content"])
    return count

def ask(question):
    global history
    if len(question) > 0:
        history.append({"role":"user","content":question})
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=history)
    reply = response["choices"][0].message.content
    history.append({"role":"assistant","content":reply})
    return reply

def getInput():
    global prefix
    global prefix_tr
    global history
    global postfix
    postfix = ""
    user_input = input(f"You[{count()}]: ")
    if(user_input.startswith("1")):
        user_input = "counts to ten."
        
    elif(user_input.startswith("2")):
        history = []
        print("History cleared.")
        return ""
    elif(user_input.startswith(">")):
        prefix = "write a python program that "
        prefix_tr = "bana bir python programı yaz şu şekilde:"
        postfix = make_shorter
        user_input = user_input[1:]

    if user_input == "exit":
        exit()
    return user_input


def chat():
    global history
    global prefix
    global prefix_tr
    global postfix
    user_input = ""
    loop = False
    
    prefix_tr = ""
    while True:
        prefix = ""
        postfix = ""
        if not loop:
            user_input = getInput()
            if user_input == "":
                continue
            reply = ask(prefix + user_input + postfix)
        else:
            reply = ask("")
        user_input = ""
        print(f"Assistant[{count()}]: ", reply)
        code = get_code(reply)
        loop = False
        if code:
            try:
                #print("Executing code: ", code)
                exec(code)
            except Exception as e:
                print(e)
                reply = ask(str(e))
                print("Assistant: ", reply)
                loop = True
                

chat() 