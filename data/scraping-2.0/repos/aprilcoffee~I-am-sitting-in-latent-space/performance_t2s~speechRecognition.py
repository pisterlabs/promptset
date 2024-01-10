import sounddevice as sd
import soundfile as sf
import os
import config
import openai
openai.api_key = config.openai_api_key
import io
from textGenerator import TextGenerator
from deviceChannels import list_input_devices
from sendingOSC import sendOSCtoVisual_question,sendOSCtoMax_answer

generator_response = TextGenerator("you are a beach tourism agency, answer only with one sentence",role='agent')    

class AudioRecorder:
    def __init__(self, sample_rate=44100, duration=15, output_folder="recordings"):
        self.sample_rate = sample_rate
        self.duration = duration
        self.output_folder = output_folder
        self.is_recording = False
        self.audio_data = None
        os.makedirs(output_folder, exist_ok=True)

                               
    def set_recording_device(id):
        sd.default.device = id

    def start_recording(self):
        if not self.is_recording:
            recording_id,channel = list_input_devices()
            sd.default.device = recording_id
            #output_id,channel = recorder.list_output_devices()
            print("Starting recording...")
            self.is_recording = True
            self.audio_data = sd.rec(int(self.sample_rate * self.duration), samplerate=self.sample_rate, channels=1, blocking=False)


    def stop_recording(self):
        if self.is_recording:
            print("Stopping recording...")
            self.is_recording = False
            sd.stop()
            output_file_name = "recorded_audio.wav"
            output_file_path = os.path.join(self.output_folder, output_file_name)
            sf.write(output_file_path, self.audio_data, self.sample_rate)
            print(f"Audio saved to {output_file_path}")
            transcript = ''
            with io.open(output_file_path, 'rb') as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en").text
                print(f"transcription:{transcript}")
                #return transcript

            sendOSCtoVisual_question(transcript)



            response,filepath,duration_seconds = generator_response.speechGPT(transcript, 0)
            sendOSCtoMax_answer(0,response,filepath)
            
            # Add your transcription code here

        

if __name__ == "__main__":
    pass
    #recorder = AudioRecorder(sample_rate=44100, duration=10, output_folder="recordings")
    #recorder.record_and_save(output_file_name="my_recording.wav")


           