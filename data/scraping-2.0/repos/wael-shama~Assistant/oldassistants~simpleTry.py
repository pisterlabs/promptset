import openai
import whisper

class AudioProcessor:
    def __init__(self):
        # Set the OpenAI API key
        openai.api_key = "<your_openai_api_key>"

        # Set up whisper
        self.transcriber = whisper.models.SileroTranscriber('en')
    
    def process_audio_file(self, audio_file_path):
        # Process audio file using whisper
        audio_data, sample_rate = whisper.load(audio_file_path)
        text = self.transcriber.transcribe(audio_data, sample_rate)

        # Generate text using OpenAI's GPT-3
        prompt = f"Transcription of audio file: {text}"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # Return the generated text
        return response.choices[0].text

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    audio_file_path = "<path_to_audio_file>"
    text = audio_processor.process_audio_file(audio_file_path)
    print(text)