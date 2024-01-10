import openai
##
import speech_recognition as sr
# from gtts import gTTS
# import pygame
from io import BytesIO
import pyttsx3
###
from PIL import Image
import yaml


image = Image.open(r'C:\Users\ashutosh.somvanshi\voice_text\pngtree-chatbot-icon-chat-bot-robot-png.png')

openai.api_key='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'

####################################################################################################
# Function for using GPT

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
     )
    return response.choices[0].message["content"]


def is_question_relevant(question):
    relevant_keywords = ["crude oil", "price", "production", "demand", "supply","Future"]
    question_lower = question.lower()
    for keyword in relevant_keywords:
        if keyword in question_lower:
            return True
    return False

# file =open(r"C:\Users\ashutosh.somvanshi\PVC_Trend_Analysis\Gpt_Data_understanding Capability\Table_data.yaml", 'r')
file =open(r"C:\Users\ashutosh.somvanshi\PVC_Trend_Analysis\Gpt_Data_understanding Capability\Review_data.yaml", 'r')
yaml_data = yaml.safe_load(file)


def get_answers_few_shot_approach_(question): 
    
    prompt = f""" 
        Your task is to answer a question based on a tabular data \
        tabular data is followed by tripple back ticks  ```{yaml_data}```  \ 
        and question is follwed by double qoutes "{question}" \ 
        while answering please do not repeat the question \
        if the question is not related to data give 'It is nor realevant to data ask something else \
        answer in 50 words or less \    
    
    """ 
    
    
    answer = get_completion(prompt)
    return answer
        
    
    
####################################################################################################   
# Function used to read speak
####################################################################################################

            
       

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    
     
    

def recognize_speech():
    r = sr.Recognizer()
    mic = sr.Microphone()
    lang_code = "en-US" #input("Enter language code (e.g. en-US, hi-IN): ")
    with mic as source:
        print("Ask your question:")
        print("Listening...")
        # r.adjust_for_ambient_noise(source)
        audio = r.listen(source,phrase_time_limit=10) #will only listen and strore in audio variable
        try:
            text = r.recognize_google(audio, language=lang_code)
            print(f"You Said : " +{text})
            return text 
        except sr.UnknownValueError:
            print("Could not understand audio, Please try speaking again")  
            return ("Could not understand audio, Please try speaking again")





transcript = str(input("Ask your question: "))
def main():
    Answer_s = ""       
    # transcript = recognize_speech()
    print(f"Question: {transcript}")
    if transcript == "Could not understand audio, Please try speaking again":
        speak(str(transcript))
        Answer_ ="Please try speaking again"
        speak(str(Answer_))
        Answer_s=Answer_
        print(f"Answer: {Answer_s}")
    
    else  :
        speak(str(transcript))
        answer__=  get_answers_few_shot_approach_(transcript)
        speak(str(answer__))
        Answer_s = answer__
        print(f"Answer: {answer__}")
    
    




if __name__ == "__main__":
    main()

