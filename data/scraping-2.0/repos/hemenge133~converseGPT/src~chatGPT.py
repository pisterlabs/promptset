from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from initChat import initChat

# load_dotenv() # Get OPENAI_API_KEY

memory, conversation = initChat()

# """
# Simple single-user API for now. POST at http:server-ip:5000/send_message
# """
# def chat(message):
#     response = conversation({"message": message})["text"]
#     return response

# """
# Reset the chat if the page is reloaded
# """
# def reset():
#     memory.clear()

"""
Running this module as the main function ie. `python chatGPT.py` will launch converseGPT with STT/TTS
"""
def main():
    SpeechRecognizer.adjustForBackgroundNoise()
    print("ready")
    while True:
        with noalsaerr():
            message = listen.listen()
        response = conversation({"message": message})["text"]
        with noalsaerr():
            speak.speak(response)

if __name__ == "__main__":
    from initSpeech import initSpeech
    noalsaerr, SpeechRecognizer, listen, speak = initSpeech()
    main()
