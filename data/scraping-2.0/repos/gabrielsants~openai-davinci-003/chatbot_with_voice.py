import openai as ai
import os
import pyttsx3

def chat(question,chat_log = None) -> str:
    if(chat_log == None):
        chat_log = start_chat_log
    prompt = f"{chat_log}Human: {question}\nAI:"
    response = completion.create(
        prompt = prompt, 
        model="text-davinci-003",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop = "\nHuman: ")
    return response.choices[0].text

def convertToVoice(answer):
    engine = pyttsx3.init()
    engine.setProperty('rate',150)
    #comment this line if you are using linux or if you want to use your system default voice
    engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
    engine.say(answer)
    engine.runAndWait()

if __name__ == "__main__":
    ai.api_key = 'YOUR_API_KEY'

    completion = ai.Completion()

    start_chat_log = ""

    print("\nYou\'re chatting with openai model text-davinci-003!\nType quit any time to stop the conversation. Enjoy!\n")
    question = ""
    while True:
        question = input("Me: ")
        if question == "quit":
            break
        elif question == "cls":
            if(os.name == "posix"):
                os.system('clear')
            else:
                os.system('cls')
        else:
            convertToVoice(chat(question,start_chat_log))
            print("\n")





