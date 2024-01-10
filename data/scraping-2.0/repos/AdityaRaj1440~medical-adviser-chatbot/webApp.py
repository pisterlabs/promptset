from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv
from pathlib import Path
import speech_recognition as sr


load_dotenv(Path(".env"))

openai.api_key= os.getenv('API_KEY')

app = Flask(__name__)

prompt_list= ['You will pretend to be a medical advisor that give suggestions to users based on their health conditions and severity of the health conditions in under 50 words.','\nAI: Please tell about your health condition and severity.']
condition: str= ""
severity: str= ""

def SpeakText(command):
	import pyttsx3
	# Initialize the engine
	engine = pyttsx3.init()
	engine.say(command)
	engine.runAndWait()
        
def speechText(speakCommand):
     # Initialize the recognizer
    r = sr.Recognizer()
    # use the microphone as source for input.
    with sr.Microphone() as source2:
        r.adjust_for_ambient_noise(source2, duration=0.2)
        # print("say something")
        #listens for the user's input
        
        SpeakText(speakCommand)
        # time.sleep(0.5)
        while(True):
            try:
                # input("start speaking")
                audio2 = r.listen(source2, timeout= 5)
                # input("end audio")

                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("Did you say ",MyText)
                return MyText
            except:
                SpeakText("Please say again")
                # time.sleep(0.5)

def get_api_response(prompt: str):
    text: str | None = None
    
    try:
        response: dict = openai.Completion.create(
            model= 'text-davinci-003',
            prompt= prompt,
            temperature= 0.9,
            max_tokens= 50,
            top_p= 1,
            frequency_penalty= 0,
            presence_penalty= 0.6,
            stop= [' Human:', ' AI:']
        )
        print(response)
        choices= response.get('choices')[0]
        text= choices.get('text')

    except Exception as e:
        print('ERROR', e)

    return text

def update_list(message: str, pl):
    pl.append(message)

def create_prompt(message: str, pl, separator=None):
    p_message: str= f'\nHuman: {message}'
    update_list(p_message, pl)
    if separator:
        pl = list(filter(("\nHuman: ").__ne__, pl))
        pl= pl[1:]
        prompt: str= '\n'.join(pl)
        return prompt
    prompt: str= ''.join(pl)
    return prompt

def get_bot_response(message: str, pl):
    prompt: str= create_prompt(message, pl)
    bot_response: str= get_api_response(prompt)

    if bot_response:
        update_list(bot_response, pl)
        pos: int= bot_response.find('\nAI: ')
        bot_response= bot_response[pos+5:]
    else:
        bot_response= 'Something went wrong...'
    return bot_response

@app.route('/chat/details', methods= ['POST'])
def chatDetails():
    if request.form.get('text')=='Send':
        user_input= request.form.get('message')
        response: str= get_bot_response(user_input, prompt_list)
    elif request.form.get('audio')=='Record':
        SpeakText('No')
        user_input= speechText("Please ask your question")
        response: str= get_bot_response(user_input,prompt_list)
        SpeakText(response)
    chatHistory= create_prompt("", prompt_list, '\n')
    print(prompt_list)
    # print('chatDeatails\n',chatDetails)
    return render_template('/chatResult.html',condition=request.form.get('condition'), severity=request.form.get('severity'), chatHistory=chatHistory)

@app.route('/chat-text', methods= ['POST'])
def chatText():
    condition= request.form.get('condition')
    severity= request.form.get('severity')
    global prompt_list
    prompt_list= prompt_list[0:2]
    user_input= 'health condition- '+condition+", severity- "+severity
    response: str= get_bot_response(user_input, prompt_list)
    # print(f'Bot: {response}')
    chatHistory= create_prompt("", prompt_list, '\n')
    # print("chatList\n", chatHistory)
    return render_template('chatResult.html',condition= condition, severity= severity, chatHistory=chatHistory)

@app.route('/chat-audio', methods= ['POST'])
def chatAudio():
    condition= speechText("Please tell about your health condition")
    severity= speechText("Please tell about your health severity")
    global prompt_list
    prompt_list= prompt_list[0:2]
    user_input= 'health condition- '+condition+", severity- "+severity
    response: str= get_bot_response(user_input, prompt_list)
    chatHistory= create_prompt("",prompt_list,'\n')
    SpeakText(response)
    return render_template('chatResult.html',condition=condition, severity=severity, chatHistory=chatHistory)

@app.route('/', methods=['GET','POST'])
def home():
    if(request.method=='GET'):
        return render_template('chatbotHome.html')
    else:
        if request.form.get('text')== 'TEXT CHATBOT':
            return render_template('chatText.html')
        elif request.form.get('audio')== 'AUDIO CHATBOT':
            return render_template('chatAudio.html')
        

if __name__== '__main__':
    app.run(debug=True)