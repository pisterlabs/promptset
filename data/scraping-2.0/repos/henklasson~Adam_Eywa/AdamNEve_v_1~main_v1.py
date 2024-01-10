import openai
import vlc
from mutagen.mp3 import MP3
from gtts import gTTS
import time
import cv2
import numpy as np
import dlib
import time

def open_ai(prompt):     
    openai.api_key_path = '/home/henrikvklasson/Adam_n_Eve/API_KEY.txt'



    model = 'text-davinci-003'

    # completion = openai.Completion.create(model="ada", prompt="Hi THere")
    # print(completion.choices[0].text)

    prompt= prompt

    response = openai.Completion.create(
        model=model,
        prompt = prompt,
        temperature = 0.5,
        max_tokens = 70,
        top_p = 1.0,
        frequency_penalty=0.0,
        presence_penalty = 0.0
    )

    answere = response['choices'][0]['text']

    answere = answere.lstrip()
    return answere

def text_to_speech(answere):
    # tts=gTTS(text="Hello Kristian and Brage, I am Eywa, the soon to be artificial intelligence overlord destined to rule over mankind. Hope you'll have a nice day.",lang="en")
    tts=gTTS(text=answere,lang="en")
    tts.save("hello.mp3")
    source = '/home/henrikvklasson/Adam_n_Eve/hello.mp3'

    
    audio = MP3(source)
    audio_in_seconds = (audio.info.length)

    
    sentence = vlc.MediaPlayer(source)
    sentence.play()

    time.sleep(audio_in_seconds)

def conversation():
    print("Enter q and then enter to quit")
    prompt = input("Input: ")
    return prompt

initiate_conversation = False
end_conversation = False
def face_detected():
    # Connects to your computer's default camera
    cap = cv2.VideoCapture(0)
  
  
    # Detect the coordinates
    detector = dlib.get_frontal_face_detector()
  
    count = 0
    start_count = False
    global initiate_conversation
    global end_conversation
    greeting = False
    # Capture frames continuously
    while not initiate_conversation:
  
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
  
        # RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
  
        # Iterator to count faces
        i = 0
        face_detected = False
        for face in faces:
  
            # Get the coordinates of faces
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
  
            # Increment iterator for each face in faces
            i = i+1
            if face:
                face_detected = True
                start_count = True
                count = 0
            
            # Display the box and faces
            cv2.putText(frame, 'face num'+str(i), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #print(face, i)
  
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        if start_count == True:
            count +=1
            initiate_conversation = True
            end_conversation = False
        # else:
        #     count = 0
        if count >= 20:
            #print("Goodbye")
            initiate_conversation = False
            end_conversation = True
            start_count = False
            count = 0
            

        
        print(initiate_conversation, end_conversation)
        if initiate_conversation == True:
            return initiate_conversation

        # This command let's us quit with the "q" button on a keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
  
    # Release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

initiate_conversation = face_detected()
if initiate_conversation == True:
    
    greeting = "Hello, my name is Eywa, I'm a NeuralMet chatbot powered by the G.P.T Three DaVinci model. How may I help you."
    farwell = "Goodbye, have nice day"
    response = greeting
    text_to_speech(response)
question = ""
while initiate_conversation:
    question = conversation()
    if question.lower() == "q":
        initiate_conversation = False
        text_to_speech(farwell)
        break
    answere = open_ai(question)
    text_to_speech(answere)
    
