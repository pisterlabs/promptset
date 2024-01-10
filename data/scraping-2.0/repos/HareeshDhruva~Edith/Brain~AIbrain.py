file_open = open("Data\\api.txt","r")
API = file_open.read()
file_open.close()
print(API)

from cmd import PROMPT
from imp import load_source
import openai
from dotenv import load_dotenv

openai.api_key = API
load_dotenv()
completion = openai.Completion()

def ReplyBrain(question,chat_log = None):
    File_log = open("DataBase\chat_log.txt","r")
    chat_log_template = File_log.read()
    File_log.close()
    
    if chat_log is None:
        chat_log = chat_log_template
        
    prompt = f'{chat_log}You : {question}\nEdith: '
    response = completion.create(
        model = "text-davinci-002",
        prompt=prompt,
        temperature = 0.5,
        max_tokens = 60,
        top_p = 0.3,
        frequency_penalty =0.5,
        presence_penalty =0)
    answer = response.choices[0].text.strip()
    chat_log_template_update = chat_log_template+f"\nYou : {question} \nEdith : {answer}"
    File_log = open("DataBase\chat_log.txt","w")
    File_log.write(chat_log_template_update)
    File_log.close()
    return answer


"""while True:
    kk = input ("Enter : ")
    print(ReplyBrain(kk))"""