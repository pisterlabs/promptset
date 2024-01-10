import openai
from elevenlabs import set_api_key



def initChatGptAPI():
    openai.api_key = 'API KEY'
    return openai



def initElevenLabs():
    set_api_key('API KEY')



def setBraveAPI():
    return 'API KEY'
