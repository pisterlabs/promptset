from flask import Flask, request


from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
#from flask_ngrok import run_with_ngrok
# 新增 TOKEN 進去
line_bot_api = LineBotApi('') 
#引號中加入你的Line Channel Access Token
#Add your line channel access token between ''

handler = WebhookHandler('')  
#引號中加入你的Line Channel Secret
#Add your line channel secret between ''

import openai
	
openai.api_key = "" 
#引號中加入你的OPENAPI api key
#Add your OPENAPI api key between "".
#os.getenv("OPENAI_API_KEY")


#import os
	
chat_language = "zh" #os.getenv("INIT_LANGUAGE", default = "zh")
	
MSG_LIST_LIMIT = 20 #int(os.getenv("MSG_LIST_LIMIT", default = 20))
LANGUAGE_TABLE = {
	  "zh": "哈囉！",
	  "en": "Hello!"
	}
class Prompt:
	    def __init__(self):
	        self.msg_list = []
	        self.msg_list.append(f"AI:{LANGUAGE_TABLE[chat_language]}")
	    
	    def add_msg(self, new_msg):
	        if len(self.msg_list) >= MSG_LIST_LIMIT:
	            self.remove_msg()
	        self.msg_list.append(new_msg)
	
	    def remove_msg(self):
	        self.msg_list.pop(0)
	
	    def generate_prompt(self):
	        return '\n'.join(self.msg_list)	
	
class ChatGPT:
    def __init__(self):
        self.prompt = Prompt()
        self.model = "text-davinci-003" #os.getenv("OPENAI_MODEL", default = "text-davinci-003")
        self.temperature = 0.9 #float(os.getenv("OPENAI_TEMPERATURE", default = 0))
        self.frequency_penalty = 0 #float(os.getenv("OPENAI_FREQUENCY_PENALTY", default = 0))
        self.presence_penalty = 0.6 #float(os.getenv("OPENAI_PRESENCE_PENALTY", default = 0.6))
        self.max_tokens = 240 #int(os.getenv("OPENAI_MAX_TOKENS", default = 240))
	
    def get_response(self):
        response = openai.Completion.create(
	            model=self.model,
	            prompt=self.prompt.generate_prompt(),
	            temperature=self.temperature,
	            frequency_penalty=self.frequency_penalty,
	            presence_penalty=self.presence_penalty,
	            max_tokens=self.max_tokens
	        )
        print(response['choices'][0]['text'].strip())
        print(response)
        return response['choices'][0]['text'].strip()
	
    def add_msg(self, text):
        self.prompt.add_msg(text)

chatgpt = ChatGPT()

app = Flask(__name__)
#run_with_ngrok(app)   #starts ngrok when the app is run

@app.route("/")
def hello():
	return "Hello World from Flask in a uWSGI Nginx Docker container with \
	     Python 3.8 (from the example template)"
         
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # Get user's message
    user_message = event.message.text
    chatgpt.add_msg(f"HUMAN:{user_message}?\n")

    reply_msg = chatgpt.get_response()
    
    
    print(reply_msg)
    #response = chatbot.get_chat_response((user_message), output="text")
    #print(response) 
    # Get opengpt's response
    #openai_response = response["message"]
    # Send response to user
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_msg)
    )

