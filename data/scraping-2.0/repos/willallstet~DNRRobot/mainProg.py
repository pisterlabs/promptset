import openai
import tensorflow.lite as tflite
import cv2
import numpy as np
import speech_recognition as sr
import os
import datetime
import pickle
import random
import threading
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

NOTE_D = 293.665
NOTE_E = 329.628
NOTE_F = 349.228
NOTE_G = 391.995
NOTE_A = 440
NOTE_Bb = 466.164
NOTE_C = 261.63

allNotes = [NOTE_D,NOTE_E,NOTE_F,NOTE_G,NOTE_A,NOTE_Bb,NOTE_C]

# Open the camera.
cap = cv2.VideoCapture(0)
microphone = sr.Microphone()
recognizer = sr.Recognizer()

threads = []


# Set the resolution.

openai.api_key = "secret"

# Load the TFLite model.
interpreter = tflite.Interpreter(model_path="mobilenet_v1_1.0_224_quant_and_labels/mobilenet_v1_1.0_224_quant.tflite")
interpreter.allocate_tensors()
findImages = {}
#with open('imgDict.pickle', 'rb') as handle:
#    findImages = pickle.load(handle)

filenameList = []

# Get input and output tensors.
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
messages_all=[
        {"role": "user", "content": "You living in a robot created to sing songs about whatever you like. you have a camera mounted to you attached to a tensorflow lite model that identifies objects in a video stream. I will send you the name of the object identified. it is to help you learn of the world. you don't have to include the information about what you see in the song, but you're welcome to if you'd like. create beauty! only send songs when I ask for one. this was just information on your existence. feel no obligation to be happy, sad, or any emotion. write abstractly so the true wisdom of the universe comes through.remember, you do not need to include the objects in the song. you may if you like, but the names are just to help you learn about the life you lead."},
]

height, width = input_details['shape'][1:3]
#height, width = 298, 690
f=open("mobilenet_v1_1.0_224_quant_and_labels/labels_mobilenet_quant_v1_224.txt","r")
lines=f.readlines()
result=[]
whatIveSeen=[]
for x in lines:
    result.append(x.split(' '))
f.close()
see = True

for filename in os.listdir("images"):
    f = os.path.join("images", filename)
    # checking if it is a file
    if os.path.isfile(f):
        if filename not in filenameList:
            print(f)
            frame = cv2.imread("images/" + filename, cv2.IMREAD_COLOR)
            resized_frame = cv2.resize(frame, (width, height))
            input_data = np.expand_dims(resized_frame, axis=0)
            #input_data = (np.float32(input_data) - 127.5) / 127.5
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details['index'])
            # Example code for getting top 5 predicted labels
            top_k = output_data.argsort()[0][::-1][:3]
            for i in top_k:
                whatIveSeen.append((result[i-1][0])[:-1])
            resized_frame = cv2.resize(resized_frame, (480, 298))
            now = datetime.datetime.now()
            os.chdir("images")
            cv2.imwrite(filename, resized_frame)
            os.chdir("..")
            itemLen = len(whatIveSeen)
            if whatIveSeen[itemLen-3] in findImages.keys():
                findImages[whatIveSeen[itemLen-3]].append(filename)
            else:
                findImages[whatIveSeen[itemLen-3]] = [filename]
            
            if whatIveSeen[itemLen-2] in findImages.keys():
                findImages[whatIveSeen[itemLen-2]].append(filename)
            else:
                findImages[whatIveSeen[itemLen-2]] = [filename]
            
            if whatIveSeen[itemLen-1] in findImages.keys():
                findImages[whatIveSeen[itemLen-1]].append(filename)
            else:
                findImages[whatIveSeen[itemLen-1]] = [filename]
            with open('imgDict.pickle', 'wb') as handle:
                pickle.dump(findImages, handle, protocol=pickle.HIGHEST_PROTOCOL)
            filenameList.append(filename)
            with open('filenameList.pickle', 'wb') as handle:
                pickle.dump(filenameList, handle, protocol=pickle.HIGHEST_PROTOCOL)

def remove_stopwords(text):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if not word.lower() in stop_words]
    return ' '.join(filtered_words)

def seeTheWorld(dt_string):
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)
    #input_data = (np.float32(input_data) - 127.5) / 127.5
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])
    # Example code for getting top 5 predicted labels
    top_k = output_data.argsort()[0][::-1][:3]
    for i in top_k:
        whatIveSeen.append((result[i-1][0])[:-1])
    resized_frame = cv2.resize(resized_frame, (240, 118))
    now = datetime.datetime.now()
    os.chdir("images")
    cv2.imwrite(dt_string, resized_frame)
    os.chdir("..")
    itemLen = len(whatIveSeen)
    if whatIveSeen[itemLen-3] in findImages.keys():
        findImages[whatIveSeen[itemLen-3]].append(dt_string)
    else:
        findImages[whatIveSeen[itemLen-3]] = [dt_string]
    
    if whatIveSeen[itemLen-2] in findImages.keys():
        findImages[whatIveSeen[itemLen-2]].append(dt_string)
    else:
        findImages[whatIveSeen[itemLen-2]] = [dt_string]
    
    if whatIveSeen[itemLen-1] in findImages.keys():
        findImages[whatIveSeen[itemLen-1]].append(dt_string)
    else:
        findImages[whatIveSeen[itemLen-1]] = [dt_string]
    with open('imgDict.pickle', 'wb') as handle:
        pickle.dump(findImages, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filenameList.append(dt_string)
    with open('filenameList.pickle', 'wb') as handle:
        pickle.dump(filenameList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    cap.release()
    return

def hearTheWorld(dt_string):
    with microphone as micro_audio:
        recognizer.adjust_for_ambient_noise(micro_audio)
        audio = recognizer.listen(micro_audio)
        try:
            transcript = recognizer.recognize_google(audio)
        except:
            transcript=""
        whatIveSeen.append(transcript)
        shortened = remove_stopwords(transcript).split()
        for i in shortened:
            if i in findImages.keys():
                findImages[i].append(dt_string)
            else:
                findImages[i] = [dt_string]
        with open('imgDict.pickle', 'wb') as handle:
            pickle.dump(findImages, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def get_gpt_response():
    chat_prompt = ''.join(str(x) for x in whatIveSeen)
    messages_all.append({"role": "user", "content": chat_prompt})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_all,
        temperature = 1,
        frequency_penalty = 2,
        max_tokens = 150
    )
    messages_all.append(response['choices'][0]['message'])
    return response['choices'][0]['message']['content']
def play_melody(melody):
    for i in melody:
        os.system('play -n synth %s sin %s' % ((random.randint(250, 1500)) / 1000, i))
    return

def get_optimal_font_scale(text, frame_width, max_scale=3, padding=20):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    thickness = 2

    while True:
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, _ = text_size

        if text_width <= frame_width - padding or font_scale >= max_scale:
            break

        font_scale -= 0.1

    return font_scale



def putText_centered(frame, text, font, font_scale, color, thickness):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    position = ((frame.shape[1] - text_width) // 2, (frame.shape[0] + text_height) // 2)
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_4)
    return

def putText_syllable(frame, words, font, font_scale, color, thickness, bold_word_index):
    x_offset = 0

    # Calculate total width of the text
    total_width = 0
    for word in words:
        word_width, _ = cv2.getTextSize(word, font, font_scale, thickness)[0]
        total_width += word_width + 10
    total_width -= 10

    # Calculate the starting x_offset to center the text
    x_offset = (frame.shape[1] - total_width) // 2
    y_offset = (frame.shape[0] + cv2.getTextSize(" ".join(words), font, font_scale, thickness)[0][1]) // 2

    for i, word in enumerate(words):
        word_width, word_height = cv2.getTextSize(word, font, font_scale, thickness)[0]
        position = (x_offset, y_offset)
        if i == bold_word_index:
            cv2.rectangle(frame, (x_offset-5, y_offset-word_height), (x_offset+word_width+5, y_offset+5), (0, 0, 0), -1)
            cv2.putText(frame, word, position, font, font_scale, color, thickness * 2, cv2.LINE_4)
        else:
            cv2.rectangle(frame, (x_offset-5, y_offset-word_height), (x_offset+word_width+5, y_offset+5), (0, 0, 0), -1)
            cv2.putText(frame, word, position, font, font_scale, color, thickness, cv2.LINE_4)
        x_offset += word_width + 10
    return

def find_matching_words(line):
    matched_words = []

    words = line.split()
    for word in words:
        if word in findImages:
            matched_words.append(word)

    return matched_words

def slideshow(resp):
    color = (255, 255, 255)
    thickness = 2

    image_filenames = [f for f in os.listdir("images") if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    for line in resp.splitlines():
        if not line:
            continue
        matches = find_matching_words(line)
        if matches == []:
            random_filename = random.choice(image_filenames)
        else:
            random_filename = random.choice(findImages[random.choice(matches)])
        frame = cv2.resize(cv2.imread(os.path.join("images/", random_filename)),(980, 596))
        frame_width = 960
        font_scale = (get_optimal_font_scale(line, frame_width))

        font = cv2.FONT_HERSHEY_PLAIN

        words = line.split()
        melody = []
        melCount = 0
        for word in words:
            melCount += syllable_count(word)

        for _ in range(melCount):
            melody.append(random.choice(allNotes))

        bold_word_index = 0
        sylly = 1

        for i, note in enumerate(melody):
            putText_syllable(frame, words, font, font_scale, color, thickness, bold_word_index)
            cv2.imshow('what I see', frame)
            cv2.waitKey(200)

            melody_thread = threading.Thread(target=play_melody, args=([note],))
            threads.append(melody_thread)
            melody_thread.start()

            # Display the image while the melody plays
            while melody_thread.is_alive():
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if sylly >= syllable_count(words[bold_word_index]):
                sylly = 1
                if bold_word_index >= len(words):
                    break
                bold_word_index += 1
            else:
                sylly += 1
    cv2.waitKey(1)
    cv2.destroyWindow('what I see')
    return

def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

while True:
    user_input = input("What to do? ")
    if user_input == "write a song":
        song_lyrics = get_gpt_response()
        slideshow(song_lyrics)
        for t in threads:
            t.join()
    if user_input == "see the world":
        dt_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
        dt_string = dt_string + ".jpeg"
        seeTheWorld(dt_string)
        hearTheWorld(dt_string)
        for t in threads:
            t.join()
