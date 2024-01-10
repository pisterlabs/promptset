import models
import torch
import data_creation
import os
from dotenv import load_dotenv
from data_creation import bpms, feels, keys, modes
import data_config
from prediction_from_file import BPM_MODEL_PATH, FEEL_MODEL_PATH, MAJOR_KEY_MODEL_PATH, MINOR_KEY_MODEL_PATH, MODE_MODEL_PATH, SECONDS_PER_CHOP, SAMPLE_RATE, NUM_SAMPLES, TEMP_NAME, INT_FOLDER
# import pyaudio
import wave
import time
from prediction_from_file import INT_FOLDER, process_song_and_make_preds
import librosa
import openai
from server import TEMP_FILE

REC_NAME = "temp_rec.wav"

TEMP_REC_PATH = os.path.join(INT_FOLDER, TEMP_FILE)
# def collect_audio():
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=pyaudio.paInt16, channels=1, rate=data_config.SAMPLE_RATE, input=True, frames_per_buffer=1024)
#     frames = []
#     seconds_needed = data_creation.SECONDS_PER_CHOP + 2
#     start_time = time.time()
#     curr_time = time.time()
#     try:
#         while True and seconds_needed >= curr_time - start_time: # it'll go for max 10 seconds (we give it 2 seconds of padding just because)
#             data = stream.read(1024)
#             frames.append(data)
#             print(f"seconds remaining: {(start_time+seconds_needed-curr_time):.1f}")
#             curr_time = time.time()
#     except KeyboardInterrupt:
#         pass
#     if curr_time - start_time < seconds_needed:
#         print("recording prematurely stopped, no processing done")
#         return
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()
#     sound_file = wave.open(TEMP_REC_PATH, "wb")
#     sound_file.setnchannels(1)
#     sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
#     sound_file.setframerate(data_config.SAMPLE_RATE)
#     sound_file.writeframes(b''.join(frames))
#     sound_file.close()
def output_bpm():
    filename = TEMP_REC_PATH
    signal, sr = librosa.load(filename)
    tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=sr)
    # print(f"BPM: {tempo:.2f}")
    return int(tempo)
def ask_gpt(bpm, key, mode, feel):
    load_dotenv()
    bpm = int(bpm)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"give me 3 tips (one sentence each) to create a good sounding, {feel}, {bpm} bpm, trap song in the key of {key} {mode}. Answer in this exact format with NO OTHER TEXT: 1. Tip 1 2. Tip 2 3. Tip 3"}])
    return completion.choices[0].message.content
def run_inference():
    bpm_model = models.BPM_Predictor(1, len(bpms))
    bpm_sd = torch.load(BPM_MODEL_PATH)
    bpm_model.load_state_dict(bpm_sd)
    feel_model = models.Feel_Predictor(1, len(feels))
    feel_sd = torch.load(FEEL_MODEL_PATH)
    feel_model.load_state_dict(feel_sd)
    major_key_model = models.Key_Predictor(1, len(keys))
    major_key_sd = torch.load(MAJOR_KEY_MODEL_PATH)
    major_key_model.load_state_dict(major_key_sd)
    minor_key_model = models.Key_Predictor(1, len(keys))
    minor_key_sd = torch.load(MINOR_KEY_MODEL_PATH)
    minor_key_model.load_state_dict(minor_key_sd)
    mode_model = models.Mode_Predictor(1)
    mode_sd = torch.load(MODE_MODEL_PATH)
    mode_model.load_state_dict(mode_sd)
    # collect_audio() # this should be done by react, we're testing API calls for now
    song_tempo, song_key, song_mode, song_feel, gpt_message = "", "", "", "", ""
    if os.path.exists(TEMP_REC_PATH):
        tempo = output_bpm()
        bpm, feel, key, mode = process_song_and_make_preds(TEMP_REC_PATH, bpm_model, feel_model, major_key_model, minor_key_model, mode_model)
        # note that the bpm from this call is not what's returned or displayed, for that we use tempo from prev line
        # os.remove(TEMP_REC_PATH)
        if bpm is not None and feel is not None and key is not None:
            # print(f"BPM: {bpms[int(bpm)]} | Key: {keys[int(key)]} | Mode: {modes[int(mode)]} | Feel: {feels[int(feel)]}")
            # print(f"Generating tips to start the beat... ")
            message = ask_gpt(tempo, keys[int(key)], modes[int(mode)], feels[int(feel)])
            # print(message)
            song_tempo = str(int(tempo))
            song_feel = str(feels[int(feel)])
            song_key = str(keys[int(key)])
            song_mode = str(modes[int(mode)])
            gpt_message = message
        # else:
        #     print("there was an issue in processing")
    return song_tempo, song_feel, song_key, song_mode, gpt_message
if __name__ == "__main__":
    tempo, feel, key, mode, message = run_inference()
    print(f"{tempo}|{feel}|{key}|{mode}|{message}")
    # if any of the above are "" then there was a problem