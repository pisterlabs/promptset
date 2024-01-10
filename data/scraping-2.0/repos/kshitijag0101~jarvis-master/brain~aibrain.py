#open ai
fileopen=open("data\\api.txt")
api=fileopen.read()
fileopen.close()
print(api)


import openai
from dotenv import load_dotenv



openai.api_key=api
load_dotenv()
completion= openai.Completion()

def replybrain(question,chat_log=None):
    filelog=open("database\chat_log.txt","r")
    chat_log_template= filelog.read()
    filelog.close()

    if chat_log is None:
        chat_log=chat_log_template
    prompt=f'{chat_log} You : {question}\n jarvis : '
    response=completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=0.3,
        frequency_penalty=0.5,
        presence_penalty=0
        )
    ans=response.choices[0].text.strip()
    chat_log_template_update=chat_log_template+ f"\n You : {question} \nJarvis : {ans}"
    filelog=open("database\chat_log.txt","w")
    filelog.write(chat_log_template_update)
    filelog.close()
    return ans


