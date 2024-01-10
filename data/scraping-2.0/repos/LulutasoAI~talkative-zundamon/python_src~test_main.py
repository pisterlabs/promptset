from tts_handler import TTSHandler
from openai_functions import voiceVoxHandler
voice_vox = voiceVoxHandler()
tts = TTSHandler()
tts.speak("会話を開始するのだ。")
response = voice_vox.callChatGPT("こんにちは")
print(response)
tts.speak(response)