from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
# from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory

import openai
from audio_recording import *
from audio_output import *
import argparse
import json



def main():
    # read the configuration from json file, and put the configuration into args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='args_private.json')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    key=config["key"]
    openai.api_key = key 
    agent_name_list=config["agent_name"]
    warning_list=["stop","end","quit","exit"]
    tts=TextToSpeech()
    # start the conversation
    while True:
        chat = ChatOpenAI(temperature=0.1,openai_api_key=key,model_name="gpt-3.5-turbo")
        memory=ConversationSummaryBufferMemory(llm=chat, max_token_limit=2000)
        conversation = ConversationChain(
        llm=chat, 
        verbose=True, 
        memory=memory,)
        audio_alarm()
        with open("alarm.wav", "rb") as audio_file:
            message_json= openai.Audio.transcribe("whisper-1", audio_file)
        human_message=message_json["text"]
        if any(word in human_message for word in agent_name_list):
            text_start="hello, I am here, how can I help you?"
            print("AI said: {}".format(text_start))
            tts.speak(text_start)
            while True:
                audio_record()
                with open("recordfile.wav", "rb") as audio_file:
                    message_json= openai.Audio.transcribe("whisper-1", audio_file)
                human_message=message_json["text"]
                print("You said: {}".format(human_message))
                # if the message contains the keywords in warning_list, then stop the conversation
                if any(word in human_message for word in warning_list):
                    text="Thank you for using me, see you next time"
                    print("AI said: {}".format(text))
                    tts.speak(text)
                    
                    break
                texts = conversation.predict(input=human_message)
                print("AI said:{}".format(texts))
                tts.speak(texts)


if __name__ == '__main__':
    main()
    
                
