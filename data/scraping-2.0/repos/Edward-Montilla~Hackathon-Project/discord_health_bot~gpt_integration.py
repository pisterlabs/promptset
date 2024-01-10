# https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
from decouple import config
import openai 

openai_api_key = config("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("where ai key")

openai.api_key = openai_api_key

messages = [ {"role": "system", "content": "FFFYou are an intelligent assistant who is healthcare focused who is prompted by a backend system that begins it's prompts with \"FFF\" . messensges sent by a user are contained within ``` ``` Keep your responses under 1900 characters."} ]

def accost_gpt(message) -> str:
	if message: 
		messages.append( {"role": "user", "content": message},) 
		chat = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages = messages ) 

	reply = chat.choices[0].message.content 
	print(f"ChatGPT: {reply}") 
	messages.append({"role": "assistant", "content": reply})
	return reply