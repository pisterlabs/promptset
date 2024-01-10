```python
import openai

class WhisperASR:
    def __init__(self):
        self.asr = openai.Whisper('your-whisper-api-key')

    def listen(self):
        # This function will listen for user commands after the wake word is detected
        # The actual implementation will depend on the specific audio input library you are using
        # For example, if you are using PyAudio, you might start a new stream here
        # and continuously read from it in a loop
        pass

    def convert_speech_to_text(self, audio_data):
        # This function will convert the audio data into text using the Whisper ASR API
        response = self.asr.recognize(audio_data)
        return response['text']

    def get_user_command(self):
        # This function will listen for a user command and then convert it into text
        audio_data = self.listen()
        command = self.convert_speech_to_text(audio_data)
        return command
```