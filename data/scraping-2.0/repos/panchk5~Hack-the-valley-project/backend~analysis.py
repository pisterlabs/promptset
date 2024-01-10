# import openai
from decouple import config
import openai
from vision import detect_text
openai.api_key = config('API_KEY_1') # API_KEY_1 is the key for the GPT-3.5-turbo model
def chat():
    language = "python3" # make way to choose language using a menu or something
    messages = [ {"role": "system", "content": "You are a helpful assistant who explains what the code is doing, do not provide feedback, the language used is" + language} ] 
    image = False
    while True:
        if image == False:
            message = detect_text()
        else:
            message = input("You: ") 
        if message == "-1": # exit 
            break
        if message: 
            messages.append( 
                {"role": "user", "content": message}, 
            ) 
            chat = openai.ChatCompletion.create( 
                model="gpt-3.5-turbo", max_tokens = 60, messages=messages
            ) 
        reply = chat.choices[0].message.content 
        print(reply) 
        messages.append({"role": "assistant", "content": reply}) 
        image = True
    return [reply, message]
    