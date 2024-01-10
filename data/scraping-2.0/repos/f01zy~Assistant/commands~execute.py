import os
import random
import webbrowser
import pathlib

# import openai

from .say import say

def execute(cmd):
        BASE_DIR = pathlib.Path().resolve()
    
    # if purpose == "cmd":
        if cmd == "yandex":
            os.system(f"{BASE_DIR}/applications/Yandex.lnk")
            say(f"ok{random.randint(1 , 4)}")
            
        elif cmd == "excellent":
            say("thanks.wav")
            
        elif cmd == "youtube":
            url = "https://youtube.com/"
            webbrowser.open(url)
            
        elif cmd == "VS Code":
            os.system(f"{BASE_DIR}/applications/Code.lnk")
            say(f"ok{random.randint(1 , 4)}")
            
        elif cmd == "figma":
            os.system(f"{BASE_DIR}/applications/Figma.lnk")
            say(f"ok{random.randint(1 , 4)}")
            
    # elif purpose == "openai":
    #     pass