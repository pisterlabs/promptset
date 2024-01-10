import asyncio
import sounddevice as sd
from scipy.io.wavfile import write
import openai
import threading
class liveTranscription():
    def __init__(self):
        self.fs = 44100  # Sample rate
        self.seconds = 3  # Duration of recording
        i = 0
        self.res = []
        self.transcription_threads = []

    def transcribe_in_thread(self, filename):
        audio_file = open(filename, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        file_name = "my_text_file.txt"
        with open(file_name, "a") as text_file:
            text_file.write(transcript.text)
        # print(transcript.text)
        # post call here
        self.res.append(transcript.text)

    def record_and_transcribe(self, i):
        # Record audio
        myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=1)
        sd.wait()  # Wait until recording is finished
        output_file = 'output' + str(i) + '.wav'
        write(output_file, self.fs, myrecording)  # Save as WAV file

        # Start a thread for transcription
        transcription_thread = threading.Thread(target=self.transcribe_in_thread, args=(output_file,))
        transcription_thread.start()
        self.transcription_threads.append(transcription_thread)

    async def main(self):
        i = 0
        tasks = []

        while i < 60:
            print(i)
            self.record_and_transcribe(i)
            i += 1
            
        # Continue recording or performing other tasks here
        print(self.res)
        return self.res

if __name__ == "_main_":
    obj = liveTranscription()
    ret = asyncio.run(obj.main())
    # Wait for all transcription threads to finish
    for thread in obj.transcription_threads:
        thread.join()
    print("".join(obj.res))