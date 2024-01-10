import openai
import sounddevice as sd
import soundfile as sf
import os
import threading
from pathlib import Path
from api_manager import *
import text_voice_converter as tvc
import random


google_ai_api_key = open_api_key_file(GOOGLE_AI_API_KEY_PATH)
open_ai_api_key = open_api_key_file(OPENAI_API_KEY_PATH)

# Set your OpenAI API key here
openai.api_key = open_ai_api_key

VOICE_FOLDER = "voices"
DEFAULT_VOICE = "onyx"

stop_recording = False

ASSISTANT_NAME = "Merlin Mystique"
USER_NAME = "Sabrina"

SUCCESS_LINK = "https://drive.google.com/file/d/1-xhSU_C2lo-i1VOGJrxQ1Ioss3RbABWi/view?usp=sharing"
SOUNDS_PATH = 'sounds'
SUCCESS_SOUND_PATH = os.path.join(SOUNDS_PATH,'success_sound.mp3')

RECORD_DURATION_SEC = 10

SOLVED = False
gemini = create_gemini_client()
answers = ["little penguin", "christmas tree", "the color purple"]

initial_prompt = [
    f"""You are a wizard shop owner named {ASSISTANT_NAME}. Only talk on the behalf of {ASSISTANT_NAME}. My name is {USER_NAME}
    You are to give me riddles that she is to complete. The first should be easy, the second one hard, and the third one even harder.
    I love English, alliteration, logic, and fantasy. Remind me that I can ask for clues and hints at any time.
    Only let me advance if I answer the riddle correctly. If I answer incorrectly, I should be able to try again as many times as I want, or move on for now. So long as I complete all three ridles, I should be able to advance.
    Don't let me know the answers even if I ask. Wait for me to ask you the first question before responding.
    don't say your own name when you answer. Don't ever speak for {USER_NAME}, only talk as {ASSISTANT_NAME}.
    The first answer should {answers[0]}. The second answer should be a {answers[1]}. The third answer should be very hard, you get to pick what the answer is.
    Only once I have solved 3 riddles can you reveal the secret link {SUCCESS_LINK}.
    the rest of the conversation will be with the user."""]
conversation = initial_prompt[0]
success_prompt = f"I hope you wrote that down, as you will need it to advance. Otherwise you can start over... Just kidding, the link has been saved to your computer next to this magical script. Haha, I love you... Uhh! I mean Ben loves you! I mean... uhh... you know what I mean. Not awkward at all. Okay goodbye {USER_NAME}!"
success_audio = tvc.create_speech_from_text(success_prompt, name="success")
stop_idle_sound_flag = threading.Event()
def pick_random_waiting_sound():
    waiting_sounds = ["sounds/waiting_sound_1.mp3", "sounds/waiting_sound_2.mp3", "sounds/waiting_sound_3.mp3"]
    return random.choice(waiting_sounds)


# Modify the WAITING_SOUND_PATH assignment in your code
WAITING_SOUND_PATH = pick_random_waiting_sound()


def play_waiting_sound():
    tvc.play_audio(pick_random_waiting_sound(), stop_flag=stop_idle_sound_flag)


def conversation_loop():
    global conversation
    global SOLVED
    print(f"Waiting sound path: {WAITING_SOUND_PATH}")
    
    
    response_number = 0
    while not SOLVED:

        user_voice_file_path = tvc.record_audio(name=f"user_voice_{response_number}",duration=RECORD_DURATION_SEC)
            
        # Start the waiting sound in a separate thread
        waiting_sound_thread = threading.Thread(target=play_waiting_sound)
        waiting_sound_thread.start()

        user_text = tvc.voice_to_text(user_voice_file_path)
        print(user_text)
        conversation += f"{USER_NAME}: {user_text}\n\n"
        assistant_text = gemini.generate_content(conversation)
        
        # Stop the waiting sound thread
        stop_idle_sound_flag.set()
        waiting_sound_thread.join()
        
        print(assistant_text.text)
        conversation += f"{ASSISTANT_NAME}: {assistant_text.text}\n\n"
        agent_dialog = assistant_text.text.replace(f"{ASSISTANT_NAME}:", "")
        agent_voice_file_path = tvc.create_speech_from_text(agent_dialog, name=f"agent_voice_{response_number}")
        tvc.play_audio(agent_voice_file_path, stop_flag=None)
        
        if SUCCESS_LINK in assistant_text.text:
            SOLVED = True
            
            tvc.play_audio(success_audio, stop_flag=None)
            tvc.play_audio(SUCCESS_SOUND_PATH, stop_flag=None)
            print("Congratulations! You have solved the riddles and found the secret link!")
            with open("solved_link.txt", "w") as file:
                file.write(SUCCESS_LINK)
            print(f"The link has been saved to solved_link.txt in case you lose the link above ;)")
            pause = input("Press enter to close...")
            break


if __name__ == "__main__":
    conversation_loop()
