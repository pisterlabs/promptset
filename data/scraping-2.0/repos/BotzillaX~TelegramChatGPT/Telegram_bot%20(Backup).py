import telebot
import os
import openai
import os
from transformers import GPT2Tokenizer


#API_Key für Telegram (Muss geändert werden) / API_Key for Telegram (needs to be changed)
Your_API_KEY = "5269430439..."
#API_Key für ChatGPT (Muss geändert werden) / API_Key for ChatGPT (needs to be changed)
api_key = "sk-ApuzAyh..."
#limit einstellen / setting up a limit
price = 0.0015






#der Code / the code:





tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #should work for up to gpt4

def process_python_file(numberOfCharacters):
    current_dir = os.getcwd()

    python_file_path = os.path.abspath(__file__)

    python_file_name = os.path.splitext(os.path.basename(python_file_path))[0]
    directory_path = os.path.dirname(python_file_path)   

    documentation_file_path = os.path.join(directory_path, "documentation.txt")
    if os.path.exists(documentation_file_path):
        with open(documentation_file_path, "a+") as file:      
            file.write(numberOfCharacters + "\n")
    with open(documentation_file_path, "r") as file:
        content = file.read()
        
    tokens = tokenizer.encode(content)
    print("Total tokens:", len(tokens))
    return tokens




def openai2(test, api_key2):
    
    
    def gpt3(text, api_key3):

        openai.api_key = api_key3
        
        response = openai.Completion.create(
            engine="text-davinci-003", #falls nötig, Änderung des Modells / if needed, changing the model
            prompt=text,
                temperature=0.1,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
        )
        content = response.choices[0].text.split(".")
        # print(content)
        
        return response.choices[0].text
    
    
    if test == "AI":
      return True
    else:
      
      response = gpt3(test, api_key2)
      limitNumber = process_python_file(str(response))
      return response, limitNumber
    

bot = telebot.TeleBot(token=Your_API_KEY)
test = "AI"
@bot.message_handler(func=lambda message: "AI" in message.text)
def greet(message):
    print("es funktioniert")
    print(message.text)
    if message.text != "beenden":
        bot.register_next_step_handler(message, greet)
        
        if message.text == "AI":
            bot.send_message(message.chat.id, "This is an AI chatbot")
        else:
            novel, tokens = openai2(message.text, api_key)
            bot.send_message(message.chat.id, str(novel))
            
            bot.send_message(message.chat.id, str(len(tokens)) +" tokens have been used until now == " + str((price/1000) * len(tokens)) +"€")


    

bot.polling()
