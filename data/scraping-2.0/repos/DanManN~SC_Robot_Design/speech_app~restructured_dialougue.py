# main_file.py
#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
import pyaudio
import speech_recognition as sr
import audioop
import math
import tempfile
import os
import wave
import subprocess
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from gtts import gTTS
import pygame
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import warnings
import torch
from transformers import AutoModel, AutoTokenizer
from fuzzywuzzy import fuzz
from openai import OpenAI
import time

warnings.simplefilter("ignore")

# Constants
navigation_phrases = ['take', 'walk', 'guide']
verbal_directions_phrases = ['point', 'tell', 'find', 'give', 'how', 'explain']

# Other constants and data
number_mapping = {
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    # Add more mappings as needed
}

commands = {
    'room 1': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 7.17515230178833, y: 0.668110728263855, z: 0.0033092498779296875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room 2': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 9.780485153198242, y: -4.3226823806762695, z: 0.0046672821044921875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room 3': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 3.4306554794311523, y: -7.267694473266602, z: 0.0068721771240234375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room 4': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: -1.6239104270935059, y: 0.7523196935653687, z: 0.00223541259765625}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room 5': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 1.6131210327148438, y: 2.954784393310547, z: 0.00484466552734375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
}
nav_outputs = {
        'room 1': "I will take you to room 1",
        'room 2': "I will take you to room 2",
        'room 3': "I will take you to room 3",
        'room 4': "I will take you to room 4",
        'room 5': "I will take you to room 5",
    }
# Other functions

def preprocess_text(text, number_mapping):
    # Replace number representations with common format
    for word, number in number_mapping.items():
        text = text.replace(word, number)
    return text

def find_best_matching_command_fuzzy(recognized_text, commands, threshold=80):
    recognized_text = preprocess_text(recognized_text, number_mapping)

    best_match_cmd, best_similarity = None, 0

    for cmd in commands.keys():
        similarity = fuzz.partial_ratio(cmd, recognized_text)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_cmd = cmd

    if best_similarity > threshold:
        return best_match_cmd, best_similarity
    else:
        return None, None



def intent_classifier(utt:str):
    ##user wants to be walked
    if any(nav_phrase in utt for nav_phrase in navigation_phrases):
        print('******* IN NAVIGATION CONDITION')
        return 0
    ##user wants verbal directions
    elif any(v_phrase in utt for v_phrase in verbal_directions_phrases):
        print('******* IN verbal CONDITION')
        return 1

    ##user wants to chat
    else:
        print('******* IN CHAT CONDITION')
        return 2

def speech_output_gen(utt:str):
    # Use gTTS to convert the voice output to speech and play it
    if utt in nav_outputs:
        print("Found voice")
        response = nav_outputs[utt]
        tts = gTTS(text=response, lang='en', slow=False)
        tts.save("temp_audio.mp3")
        os.system("mpg321 temp_audio.mp3")
        print("Audio file Made")

    else:
        print('IN MISC SPeech output')
        response = "Please be sure to select a valid room, and I can walk you there or give you directions."
        tts = gTTS(text=response, lang='en', slow=False)
        tts.save("temp_audio.mp3")
        os.system("mpg321 temp_audio.mp3")
        print("Audio file Made")

def main():
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Set up audio stream parameters
    input_device_index = None
    sample_rate = 22050  # Reduced sample rate
    chunk_size = 8192  # Increased chunk size (in bytes)
    threshold_db = 60  # Adjusted threshold

    # Create a temporary audio file
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio_file_name = temp_audio_file.name
    temp_audio_file.close()

    # Open an input audio stream
    input_stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
        input_device_index=input_device_index
    )

    print("Listening...")

    # Initialize variables for voice activity detection
    audio_data = bytearray()
    speech_started = False

    # Load MiniLM model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    weight_path = "kaporter/bert-base-uncased-finetuned-squad"
    # loading tokenizer
    tokenizer = BertTokenizer.from_pretrained(weight_path)
    #loading the model
    model = BertForQuestionAnswering.from_pretrained(weight_path)

    text = "how do i get to room 4?"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print("MODEL OUTPUT: ", output)

    # Define the voice outputs dictionary
    nav_outputs = {
        'take me to room one': "I will take you to room 1",
        '2': "I will take you to room 2",
        '3': "I will take you to room 3",
        '4': "I will take you to room 4",
        '5': "I will take you to room 5",
    }


    try:
        while True:
            audio_chunk = input_stream.read(chunk_size, exception_on_overflow=False)
            audio_data.extend(audio_chunk)

            rms = audioop.rms(audio_chunk, 2)
            decibel = 20 * math.log10(rms) if rms > 0 else 0

            if decibel > threshold_db:
                if not speech_started:
                    print("Speech Started")
                    speech_started = True
            else:
                if speech_started:
                    print("Speech Ended")

                    with open(temp_audio_file_name, "wb") as f:
                        wav_header = wave.open(temp_audio_file_name, 'wb')
                        wav_header.setnchannels(1)
                        wav_header.setsampwidth(2)
                        wav_header.setframerate(sample_rate)
                        wav_header.writeframes(audio_data)
                        wav_header.close()

                    with sr.AudioFile(temp_audio_file_name) as source:
                        try:
                            transcription = recognizer.record(source)
                            recognized_text = recognizer.recognize_google(transcription)
                            if recognized_text:
                                print("Transcription: " + recognized_text)

                                intent = intent_classifier(recognized_text)

                                if intent == 0: #walk navigation
                                    best_match_cmd, best_similarity = find_best_matching_command_fuzzy(recognized_text, commands)

                                    if best_match_cmd is not None:
                                        print(f'Best match: {best_match_cmd} (Similarity: {best_similarity}%)')
                                        ros_command = commands[best_match_cmd]
                                        print(ros_command)
                                        print("trying to play speech")
                                        speech_output_gen(best_match_cmd)
                                        print("played speech")
                                        # Send the ROS message using subprocess
                                        subprocess.check_output(ros_command, shell=True, stderr=subprocess.STDOUT)

  

                                elif intent == 1: ##verbal directions
                                    #question = "How do i get from home to room 1?"  
                                    question = recognized_text                                  
                                    """
                                    input_ids = tokenizer.encode(question, context)
                                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                                    sep_idx = tokens.index('[SEP]')
                                    token_type_ids = [0 for i in range(sep_idx+1)] + [1 for i in range(sep_idx+1,len(tokens))]

                                    # Run our example through the model.
                                    out = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                                    token_type_ids=torch.tensor([token_type_ids]))

                                    start_logits,end_logits = out['start_logits'],out['end_logits']
                                    # Find the tokens with the highest `start` and `end` scores.
                                    answer_start = torch.argmax(start_logits)
                                    answer_end = torch.argmax(end_logits)

                                    ans = ' '.join(tokens[answer_start:answer_end])
                                    """
                                    client = OpenAI(api_key="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")

                                    response = client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant who will provide directions between two points based off the description."},
                                            {"role": "user", "content": "Description: The structure we're currently occupying comprises five distinct rooms, along with a designated 'home' base where I can retreat to after guiding a user to their intended destination. Let me provide you with a more detailed overview of each room:\n* Home Location:\n    * Positioned at the corner closest to the left wall of the room, opposite the exterior glass wall. This serves as the central point of operations, conveniently accessible after assisting users in reaching their destinations.\n* Room 1:\n    * Located to the right of the television, at the corner where the interior walls meet. This space offers a cozy atmosphere, with a view extending towards the central area of the room.\n* Room 2:\n    * Situated near the left-hand corner when facing the interior entrance door. This room features a welcoming ambiance, with natural light streaming in from adjacent windows, creating an inviting environment.\n* Room 3:\n    * Found adjacent to the right-hand corner when facing the interior entrance door. This room boasts a strategic layout, offering a balance of privacy and accessibility.\n* Room 4:\n    * Positioned by the external exit, closest to the cabinets. This room is conveniently located for quick access to outdoor areas and is characterized by its proximity to functional storage spaces.\n* Room 5:\n    * Situated along the same exterior wall but at the far-right end. This room enjoys a quieter setting compared to the others, with a view extending along the exterior of the building."},
                                            {"role": "user", "content": question}
                                        ]
                                    )
                                    ans=list(list(list(response.choices[0])[2][1])[0])[1]
                                    if ans:
                                        print('Predicted answer:', ans)
                                    else:
                                        print("I have no answer")
                                        ans=" I have no answer"

                                    tts = gTTS(text=ans, lang='en', slow=False)
                                    tts.save("temp_audio.mp3")
                                    os.system("mpg321 temp_audio.mp3")
                                    print("Audio file Made")
                                else:   #misc
                                    speech_output_gen(recognized_text)  


                                # Play the audio file using pygame
                                pygame.mixer.init()
                                pygame.mixer.music.load("temp_audio.mp3")
                                pygame.mixer.music.play()
                                
                                # Add a delay to ensure that the audio playback is completed
                                while pygame.mixer.music.get_busy():
                                    pygame.time.Clock().tick(10)

                        except sr.UnknownValueError:
                            print("No speech detected")
                        except sr.RequestError as e:
                            print(f"Could not request results; {e}")

                    speech_started = False
                    audio_data = bytearray()

    except KeyboardInterrupt:
        pass

    input_stream.stop_stream()
    input_stream.close()
    os.remove(temp_audio_file_name)
    audio.terminate()

if __name__ == "__main__":
    main()
