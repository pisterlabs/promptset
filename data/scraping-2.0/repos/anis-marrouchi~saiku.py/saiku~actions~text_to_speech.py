import os
import subprocess
import openai

class TextToSpeechAction:
    def __init__(self, agent):
        self.agent = agent
        self.name = 'text_to_speech'
        self.description = 'Converts text to speech and returns the path to the generated audio file.'
        self.parameters = [
            {
                'name': 'text',
                'type': 'string',
                'description': 'The text to be converted to speech.',
                'required': True
            },
            {
                'name': 'play',
                'type': 'boolean',
                'description': 'Play the audio file after it is generated.',
                'required': True,
                'default': True
            }
        ]

    async def run(self, args):
        text = args.get('text')
        play= args.get('play', True)
        if os.name == 'posix' and os.uname().sysname == 'Darwin':
            # For macOS, use 'say' command
            subprocess.run(['say', text])
            return f"Text spoken on macOS using Siri: {text}"
        else:
            try:
                
                speech_response = openai.Audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text
                )
                audio_buffer = speech_response['data']
                audio_file_path = 'speak.mp3'
                with open(audio_file_path, 'wb') as audio_file:
                    audio_file.write(audio_buffer)

                # Play the audio file if required
                if play:
                    if os.name == 'posix':
                        subprocess.run(['play', audio_file_path])
                    elif os.name == 'nt':
                        os.startfile(audio_file_path)

                return f"Text spoken using OpenAI and saved to: {audio_file_path}"
            except Exception as error:
                return f"Error in Text to Speech: {str(error)}"
