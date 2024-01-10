import openai
import os
from getKey import getKey

def initEnvironment():
    os.chdir("..")
    key = getKey()
    os.chdir("bot_builder")

    openai.api_key = key

    conversation = open("history.txt", "a+")
    conversation.truncate(0)
    conversation.close()