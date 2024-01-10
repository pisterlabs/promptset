import os
import queue
import openai
import sounddevice as sd
import wavio


class SpeechToTextAction:
    dependencies = ["openai", "sounddevice", "wavio"]

    def __init__(self, agent):
        self.agent = agent
        self.name = "speech_to_text"
        self.description = "Transcribe audio to text"
        self.parameters = [{"name": "audioFilename", "type": "string", "required": False}]
        self.audioFilename = 'recording.wav'

    async def init(self):
        # Initialize any other setup tasks if necessary
        pass

    async def run(self, args):
        audioFilename = args.get("audioFilename")
        await self.init()
        audioFilename = audioFilename or self.audioFilename
        if not audioFilename:
            await self.record_audio(audioFilename)
        transcription = await self.transcribe_audio(audioFilename)
        print('Transcription:', transcription)
        return transcription

    async def record_audio(self, filename):
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        with sd.InputStream(samplerate=16000, channels=1, callback=callback):
            print("Recording... Press ENTER to stop.")
            await input()  # Waits for ENTER key press
            print("Finished recording")

        q = queue.Queue()
        wavio.write(filename, q.get(), 16000, sampwidth=2)

    async def transcribe_audio(self, filename):
        
        with open(filename, 'rb') as audio_file:
            
            transcript = openai.Audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript['text']

    async def close(self):
        # Clean up resources if necessary
        pass
