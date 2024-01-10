fileopen = open("Data\\API.txt","r")
API = fileopen.read()
fileopen.close()
# print(API)

import openai
from dotenv import load_dotenv

openai.api_key = API
load_dotenv()
completion = openai.Completion()

def ReplyBrain(question, chat_log = None):
    FileLog = open("DataBase\chat_log.txt","r")
    chat_log_template = FileLog.read()
    FileLog.close()
    
    if chat_log is None:
        chat_log = chat_log_template
    
    prompt = f"{chat_log} You : {question}\n Jarvis : "
    response = completion.create(
        model = "text-davinci-002",
        prompt = prompt,
        temperature = 0.5,
        max_tokens = 60,
        top_p = 0.3,
        frequency_penalty = 0.5,
        presence_penalty = 0
    )
    answer = response.choices[0].text.strip()
    chat_log_template_update = chat_log_template + f"\nYou : {question} \nJarvis : {answer}"
    FileLog = open("DataBase\chat_log.txt", "w")
    FileLog.write(chat_log_template_update)
    FileLog.close()
    return answer

# while True:
#     reply = input("Enter: ")
#     print(ReplyBrain(reply))