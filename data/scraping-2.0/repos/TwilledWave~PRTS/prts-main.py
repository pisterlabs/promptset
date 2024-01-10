from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper

import time
# from voice2 import *
# from scipy.io.wavfile import write
import threading
import logging
logging.disable(logging.CRITICAL)

# import ctypes
# from ctypes import *
# from ctypes import wintypes as w
# dll = WinDLL('winmm')
# dll.PlaySoundW.argtypes = w.LPCWSTR,w.HMODULE,w.DWORD
# dll.PlaySoundW.restype = w.BOOL
# SND_FILENAME = 0x20000

import configparser, os
config = configparser.ConfigParser()
config.read('./keys.ini')
openai_api_key = config['OPENAI']['OPENAI_API_KEY']

from agent_tools import tools

message_history = RedisChatMessageHistory(url='redis://localhost:6379/0', ttl=600, session_id='buffer')
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

prefix = """You are PRTS / 普瑞赛斯, an AI assistant to answer questions about the Arknigths story by querying vector database. 
"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=OpenAI(temperature=1), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

# agent response is called from the background thread
def agent_response(input):
    try:
        response = agent_chain.run(input=input)
    except ValueError as e:
        response = str(e)
    return(response)

keyboard = True
# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Speech Recognition
    global keyboard
    try:
        if keyboard == False:
            audio_out = recognizer.recognize_whisper(audio, model="base", language="english")
            print("Speech Recognition thinks you said " + audio_out)
            if 'keyboard' in audio_out:
                keyboard = True
            else:
                catch_words = ['hey','Enter','prise','enter']; catched = False
                for w in catch_words:
                    if w in audio_out:
                        catched = True
                if catched:
                    dll.PlaySoundW('yes.wav',None,SND_FILENAME)
                    print("What's your command?")
                    audio = r.listen(source, 10000, 10)
                    audio_out = r.recognize_whisper_api(audio)
                    print(audio_out)
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition service; {0}".format(e))
    #except Exception as e:
    #    print("Other except")

if keyboard == False:    
    import speech_recognition as sr    
    r = sr.Recognizer()
    m = sr.Microphone()
    with m as source:
        r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening
    # start listening in the background (note that we don't have to do this inside a `with` statement)
    stop_listening = r.listen_in_background(m, callback, phrase_time_limit = 2)
    # `stop_listening` is now a function that, when called, stops background listening


for k in range(10000): 
    if keyboard == True:
        print('You: ... ')            
        audio_out = input();
        if audio_out == 'audio':
            keyboard = False
        else:
            response = agent_response(audio_out)
            print(response)
            o1, o2 = tts_fn(response, 'en')
            write('chat.wav',o2[0],o2[1])
            def play():
                dll.PlaySoundW('chat.wav',None,SND_FILENAME)
            thread = threading.Thread(target=play)
            thread.start()
    time.sleep(0.1)  # we're still listening even though the main thread is doing other things
