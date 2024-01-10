#!/usr/bin/python

"""

pip install azure-cognitiveservices-speech
pip install openai
pip install tiktoken

"""

import time
import openai
import azure.cognitiveservices.speech as speechsdk
import tiktoken
import rospy
from std_msgs.msg import Int16
import rospkg
import rospy
import actionlib
from pal_interaction_msgs.msg import TtsAction, TtsGoal


# TODO: add your keys here!

OPENAI_API_KEY = 'sk-<your key>'
AZURE_SPEECH_API_KEY = '<your key>'
AZURE_SPEECH_API_REGION = ''
RECOGNIZER_LANGUAGE = "en-US"

rospack = rospkg.RosPack()
SYSTEM_MESSAGE_FILE = f"{rospack.get_path('tiago_gpt')}/src/system_message.txt"
PRE_PROMPT_FILE = f"{rospack.get_path('tiago_gpt')}/src/pre_prompt.txt"

with open(SYSTEM_MESSAGE_FILE) as f:
	SYSTEM_MESSAGE = f.read().strip()

with open(PRE_PROMPT_FILE) as f:
	PRE_PROMPT = f.read().strip()
        
MODEL = "gpt-4"
TRIGGERWORD = "robot"

MAX_TOKENS = 7680 # 8192 - 512 arbitary choice

openai.api_key = OPENAI_API_KEY

speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_API_KEY, region=AZURE_SPEECH_API_REGION)
speech_config.speech_recognition_language = RECOGNIZER_LANGUAGE
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

transcribed_text = ''
messages=[{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": PRE_PROMPT}]

encoding = tiktoken.encoding_for_model(MODEL)
num_tokens = 0

talking = False
done = False

rate = None
gpt_marker_publisher = None
client = None

def stop_cb(evt):
    global speech_recognizer
    global done
    print('CLOSING on {}'.format(evt))
    speech_recognizer.stop_continuous_recognition()
    done = True

# Logic from https://platform.openai.com/docs/guides/chat/introduction
def num_tokens_from_message(message):
    global encoding
    num_tokens = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
    for key, value in message.items():
        num_tokens += len(encoding.encode(value))
    return num_tokens


def remove_message():
    global messages, num_tokens
    message = messages.pop(2) # always removing first real message
    num_tokens -= num_tokens_from_message(message)

# sending text to Tiago
def speech_synth(text):
    global client
    print("Saying: {}".format(text))

    goal = TtsGoal()
    goal.rawtext.text = text
    goal.rawtext.lang_id = "en_GB"

    client.send_goal(goal)
    client.wait_for_result()
   
def generate_chat_response_streaming():
    global transcribed_text, num_tokens, rate, gpt_marker_publisher
    user_message = {"role": "user", "content": transcribed_text}
    messages.append(user_message)
    num_tokens += num_tokens_from_message(user_message)
    while num_tokens > MAX_TOKENS:
        remove_message()

    transcribed_text = ""
    completion = openai.ChatCompletion.create(model=MODEL, messages=messages,  stream=True)
    entire_response = ''
    text = ''
   
    for r in completion:
        delta = r["choices"][0]["delta"]
        if "content" in delta:
            item = delta["content"]
            text += item
            if len(item.strip()) > 0:
                item_start = item.strip()[0]
                if item_start in ".:;?!": # sending to speech synth when part of the sentence is finished
                    speech_synth(text)
                    entire_response += text
                    text = ''
            
    speech_synth(text)
    entire_response += text
    text = ''
    
    assistant_message = {"role": "assistant", "content": entire_response}
    messages.append(assistant_message)
    num_tokens += num_tokens_from_message(user_message)
    messages.append({"role": "user", "content": "Did human ask you to bring an object and is the information feasible for determining which one? Only answer Y[nr] or N"})
    completion = openai.ChatCompletion.create(model=MODEL, messages=messages)
    decision = completion["choices"][0]["message"]["content"]
    message_clear = True if decision[0] == 'Y' else False

    object = decision[1:] if message_clear else None 
    object_int = None
    if message_clear:
        try: 
            object_int = int(object) # Just checking if int not much logic for handling situations where gpt is creative, gpt-4 didn't have problems with this
            rospy.loginfo(str(object_int))
            gpt_marker_publisher.publish(object_int)
            rate.sleep()
        except:
            object_int = None
            print(f"{object} cannot convert to ints")
        
    return transcribed_text, messages, message_clear, object_int

def GPT_turn():
    global talking, transcribed_text, messages
    print("\n\nGPT output:")
    gpt_answer, messages, message_clear, object = generate_chat_response_streaming()
    time.sleep(1) # to make sure it won't recognize the last sentence
    talking = False

def Azure_speech_recognizing_handler(e : speechsdk.SpeechRecognitionEventArgs) :
    global talking, transcribed_text
    if not talking:
        col_text = e.result.text.strip(' ')
        print(col_text)
        print("\n")
        transcribed_text += " " + e.result.text
        text = e.result.text.lower().replace("robert", "robot") # just a hack to make it work better
        if TRIGGERWORD in text:
            talking = True

def main_Azure_STT():
    global talking, client, transcribed_text, messages, speech_recognizer, encoding, num_tokens, rate, gpt_marker_publisher
    print("GPT is awake and listening!")
    print(f"GPT input:")

    # Tiktoken stuff
    num_tokens = 1 + num_tokens_from_message(messages[0])

    # Azure speech recognition stuff
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    speech_recognizer.recognized.connect(Azure_speech_recognizing_handler)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)
    speech_recognizer.start_continuous_recognition()

    # ROS stuff
    rospy.init_node("tiago_gpt")
    rate = rospy.Rate(10) 

    # Publisher marker id
    gpt_marker_publisher = rospy.Publisher("recognized", Int16, queue_size=0)
    rospy.sleep(2) 

    client = actionlib.SimpleActionClient('/tts', TtsAction)
    client.wait_for_server()

    while not done and not rospy.is_shutdown():
        if talking:
            GPT_turn()
            print("\n\nGPT input:")
        else:
            time.sleep(.5)    


if __name__ == "__main__":
    main_Azure_STT()
    print("GPT is going to sleep now!")
