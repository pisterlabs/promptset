import openai
from config.config import settings

def convert_speech_to_text(audio, language):
    """Convert speech to text
    """    
    def openai_whisper_speech_to_text(audio_file):
        """ Open AI - Whisper
        """
        # Retrieve Enviornment Variables
        # openai.organization = settings.OPEN_AI_ORG
        openai.api_key = settings.OPENAI_API_KEY

        try:
            transcript = openai.Audio.transcribe(
                model="whisper-1", 
                file=audio, 
                language=language)
            message_text = transcript["text"]
            return message_text
        except Exception as e:
            print("exception:", e)
            return
    price = 0
    return openai_whisper_speech_to_text(audio), price
