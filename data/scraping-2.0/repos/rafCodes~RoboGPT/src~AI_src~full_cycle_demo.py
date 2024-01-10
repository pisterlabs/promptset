# Raphael Fortuna (raf269) 
# Rabail Makhdoom (rm857) 
# Final Project Report
# Lab 403, Lab Section:  4:30pm-7:30pm on Thursdays 

from chat_conversation import openai_chat
from speech_to_text import speech_to_text
from text_to_speech import text_to_speech

class full_cycle_demo:

    def __init__(self, prompt = "You are a helpful assistant.", max_tokens = 100, debug = False):
        self.chat_instance = openai_chat(system_prompt = prompt, debug = debug, max_tokens=max_tokens)
        self.speech_instance = speech_to_text(debug = debug)
        self.text_instance = text_to_speech(debug = debug)

    def run_demo(self):
        """ run the demo """

        self.chat_instance.init_chat()

        iteration_count = 50 # number of times to run the chat cycle

        i = 0
        
        while (i < iteration_count):

            if self.speech_instance.get_speech_text():

                user_spoken_text = self.speech_instance.get_collected_text()

                # don't do anything if the user didn't say anything
                if user_spoken_text != "":
                    assistant_response = self.chat_instance.voice_chat(user_spoken_text)
                    i += 1
                    if assistant_response:
                        self.text_instance.speak_text(assistant_response)
                else:
                    print("No text collected")

        
        self.text_instance.speak_text("This is the end of the demo, thank you for using the demo today.")

if __name__ == "__main__":

    demo_instance = full_cycle_demo()
    demo_instance.run_demo()