import logging

import threading

import time

import cv2
import numpy as np
import dlib
from inputimeout import inputimeout
import vlc
from gtts import gTTS
import time


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
    while True:
  
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
            

        
        #print(initiate_conversation, end_conversation)
        

        # This command let's us quit with the "q" button on a keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
  
    # Release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

def open_ai(question):
    import openai 

    openai.api_key_path = '/home/henrikvklasson/Adam_n_Eve/API_KEY.txt'

    #If your API key is stored in a file, you can point the openai module at it 
    # with 'openai.api_key_path = <PATH>'.



    model = 'text-davinci-003'

    # completion = openai.Completion.create(model="ada", prompt="Hi THere")
#    print(completion.choices[0].text)

    prompt= question

    response = openai.Completion.create(
        model=model,
        prompt = prompt,
        temperature = 0.5,
        max_tokens = 50,
        top_p = 1.0,
        frequency_penalty=0.0,
        presence_penalty = 0.0
    )

    answere = response['choices'][0]['text']

    answere = answere.lstrip()
    print(answere)
    return answere


def thread_function(name):

    logging.info("Thread %s: starting", name)

    time.sleep(2)

    logging.info("Thread %s: finishing", name)

def tts(response):
    tts=gTTS(text=response,lang="en")
    tts.save("hello.mp3")
    source = '/home/henrikvklasson/Adam_n_Eve/hello.mp3'

    from mutagen.mp3 import MP3
    audio = MP3(source)
    audio_in_seconds = (audio.info.length)

    
    sentence = vlc.MediaPlayer(source)
    sentence.play()

    time.sleep(audio_in_seconds)

if __name__ == "__main__":

    format = "%(asctime)s: %(message)s"

    logging.basicConfig(format=format, level=logging.INFO,

                        datefmt="%H:%M:%S")


    logging.info("Main    : before creating thread")

    #x = threading.Thread(target=thread_function, args=(1,))
    face = threading.Thread(target=face_detected, args=())
    face.start()

    
    logging.info("Main    : before running thread")

    #x.start()
    
    greet = False
    response = ''
    answere = ''
    while True:
        if initiate_conversation == True and greet == False:
            response = "Hello, nice to meet you"
            tts(response)
            greet = True
        while initiate_conversation == True and greet == True:
            try:
                input = inputimeout(prompt='Enter question: ',timeout=30)
                answere = open_ai(input)
                tts(answere)
            except Exception:
                pass
            
        if end_conversation == True and greet == True:
             response = "Goodbye, have a great party"
             tts(response)
             greet = False



    logging.info("Main    : wait for the thread to finish")

    face.join()

    logging.info("Main    : all done")