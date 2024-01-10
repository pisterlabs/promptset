# import sounddevice as sd
# import numpy as np
# import wavio
# import openai
# import os

# # def record_until_silence(file_name, threshold=0.01, duration=10, samplerate=44100):
# #     audio_data = np.array([])

# #     def callback(indata, frames, time, status):
# #         nonlocal audio_data
# #         if status:
# #             print(status)
# #         audio_data = np.append(audio_data, indata)

# #     with sd.InputStream(callback=callback, channels=1, samplerate=samplerate):
# #         print(f"Recording... (Press Ctrl+C to stop)")
# #         try:
# #             sd.sleep(duration * 1000)
# #             while np.max(np.abs(audio_data)) > threshold:
# #                 audio_data = np.array([])
# #                 sd.sleep(100)
# #         except KeyboardInterrupt:
# #             pass

# #     wavio.write(file_name, audio_data, samplerate, sampwidth=2)


# # record_until_silence(r"C:\Users\Spher\OneDrive\Desktop\CS\AI\Kuebiko\audio")

# transcript = openai.Audio.transcribe("whisper-1", os.path(r"C:\Users\Spher\OneDrive\Desktop\CS\AI\Kuebiko\output22"))

# import speech_recognition as sr
# import sounddevice as sd
# import wavio
# import keyboard

# def record_audio(file_name, duration=5, samplerate=44100):
#     # Record audio
#     # while True:
#     #     if keyboard.is_pressed('o') and keyboard.is_pressed('u'):
#     #         print("hello")
#     #         break

#     audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='int16')
#     sd.wait()

#     # Save audio as WAV file
#     wavio.write(file_name, audio_data, samplerate, sampwidth=2)

#     audio_file = open(file_name, "rb")  

#     recognizer = sr.Recognizer()

#     with sr.AudioFile(audio_file) as source:
#         audio = recognizer.record(source)
#         text = recognizer.recognize_google(audio)
#         print(text)


# record_audio(r"C:\Users\Spher\OneDrive\Desktop\CS\AI\Kuebiko\output22.mp3")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_path = r"C:\Users\Spher\OneDrive\Desktop\CS\AI\Kuebiko\output22.mp3"
y, sr = librosa.load(audio_path)

print(y)
print(sr)

# Generate the mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

# Convert to decibels (log scale)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Display the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()
