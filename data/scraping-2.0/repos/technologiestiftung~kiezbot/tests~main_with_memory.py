import os
import subprocess
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from utils.gtts_synthing import synthing
from utils.characterRFID_dict import getCharacter
from utils.character_prompts import getPrompt

def makeConversation(chain):
     # Sending an empty user input first to let the AI start the conversation
    user_input = ""
    
    # greeting audio is in a subprocess in order not to block the main thread 
    subprocess.Popen(["afplay", "audio/bee_greetings.mp3"])
    
    reply = chain.predict(input=user_input)
    print(reply)

    while user_input.lower() != "q":
        user_input = input("Enter input (or 'q' to quit): ")

        if user_input.lower() != "q":
            # Play some local audio to shorten the waiting time while we wait for synthing
            subprocess.Popen(["afplay", "audio/bee_wait.mp3"])
            reply = chain.predict(input=user_input)
            print(reply)
            synthing(reply)
            play_audio("reply")

def play_audio(audio):
    os.system("afplay " + "output_gtts.mp3")

def main():
    os.system("clear")
    load_dotenv()

    characterCode = "1"
    # Voice output is currently only supported for character 1 (Bee) – to enable character selection, uncomment the following line
    # characterCode = input("Charakter auswählen (1-5) oder RFID Chip auflegen: ")
  
    prompt = getPrompt(getCharacter(characterCode))
    
    chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    chain = ConversationChain(llm=chatgpt, verbose=False, memory=ConversationBufferMemory(), prompt=prompt)
    makeConversation(chain)

if __name__ == '__main__':
    main()