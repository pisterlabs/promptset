file = open("Data\API.txt","r") #opening API.txt in reading mode which contains the secret key generated using openai
API = file.read()
file.close()
import openai #pip install openai
import dotenv  #pip install python-dotenv # used to read value in key value pair from a environment 
from dotenv import load_dotenv
openai.api_key = API
load_dotenv()
completion = openai.Completion()
# this function helps to activate ai and read and write user inputs and ai answers in chat_log.txt in Database folder
#this model can learn from its past experience using the saved data in chat_log file .
def ReplyBrain(question,chat_log=None):
    FileLog = open("Database\chat_log.txt","r") #opening chat_log filo in read mode
    chat_log_template = FileLog.read()
    FileLog.close()

    if chat_log is None :
        chat_log = chat_log_template

    prompt = f'{chat_log}You : {question}\nCbum : '
    response = completion.create(
        model = "text-davinci-002",
        prompt=prompt,
        temperature = 0.5,                 # using davinci model  
        max_tokens=60,
        top_p=0.3,
        frequency_penalty = 0.5,
        presence_penalty=0)
    
    answer = response.choices[0].text.strip()
    chat_log_template_update = chat_log_template + f'\nYou : {question} \nCbum : {answer}'
    FileLog = open("database\chat_log.txt","w")
    FileLog.write(chat_log_template_update)             # opening file in writing  mode so that we can train our model according to us 
    FileLog.close()                                     # writing both your question and answers in a text file named chat_log in database folder

    return answer

