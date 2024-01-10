import os
import openai
import config
import whisper

class Assistant:
    def __init__(self, whisper_model, openai_api_key):
        self.whisper_model = whisper_model
        self.openai_api_key = openai_api_key
        self.whisper = None
        self.openai = None

    def load_whisper(self):
        self.whisper = whisper.load_model(self.whisper_model)

    def load_openai(self, introduction=None):
        openai.api_key = self.openai_api_key
        if introduction:
            self.generate_text(introduction)

    def transcribe_audio(self, audio_file):
        if not self.whisper:
            self.load_whisper()
        return self.whisper.transcribe(audio_file)

    def generate_text(self, prompt):
        if not self.openai:
            self.load_openai()
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text


# Set the paths to the Whisper model and OpenAI API key
# print(os.environ['OPENAI_API_KEY'])
whisper_model = "base"
openai_api_key = config.OPEN_API_KEY

# Initialize the assistant
assistant = Assistant(whisper_model, openai_api_key)

# Transcribe an audio file
transcribed_text = assistant.transcribe_audio('./audios/appointment.mp3')
print('Transcribed text:', transcribed_text['text'])

# Only Text from response
# print('Transcribed text:', transcribed_text["text"])

# Generate text using OpenAI's GPT-3
# generated_text = assistant.generate_text('Write a to do action for this conversation')
generated = assistant.generate_text('This is Dr. Larrys office, Give me a summary and Write a calendar invitaton for this call: ' + transcribed_text['text'])

print('Calendar Invitation: ', generated)