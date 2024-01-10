OPENAI_API_KEY = "YOUR_GPT_KEY"
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
import pyttsx3
import speech_recognition as sr
import time

template = """
{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

output = chatgpt_chain.predict(
    human_input="""I want you to act as a Linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. Do not write explanations. Do not type commands unless I instruct you to do so. When I need to tell you something in English I will do so by putting text inside curly brackets {like this}. My first command is pwd.
    Like:
    Human: Open a pod in kubernetes of image centos:7
    Assistance: kubectl run --image centos:7"""
)
# print(output)
def hello(prompt):
    output = chatgpt_chain.predict(human_input=prompt)
    print(output)
    return(output)

def speak_and_convert_to_text(duration=5):
    engine = pyttsx3.init()

    # Ask the user to speak
    print("Listening... (Speak now)")
    engine.say("Listening... Speak now")
    engine.runAndWait()

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording... (Speak within the next 5 seconds)")
        audio = recognizer.listen(source, timeout=duration)

    try:
        print("Recognizing...")
        recognized_text = recognizer.recognize_google(audio)
        print("Recognition successful!")
        return recognized_text
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio.")
    except sr.RequestError as e:
        print(f"Error occurred while requesting results from Google Speech Recognition service: {e}")

    return None



humanvoice = speak_and_convert_to_text(duration=5)
print("You said:", humanvoice)
command = hello(humanvoice)
new_command = command.replace("`", "")
os.system(new_command)
