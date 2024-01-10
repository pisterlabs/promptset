#!/usr/bin/python3

import tkinter as tk
from tkinter import Button
import argparse
import tempfile
import queue
import sys
import os
import base64
from threading import *

import sounddevice as sd
import soundfile as sf

import whisper
import openai

from gtts import gTTS

# for the CURL post hack until there's a proper interface
import pycurl
import json
from io import BytesIO
import ffmpeg # the ugly part: it only knows webm yet

class GPTVoiceInterface:

	running = False
	local_transcription = True 

	valid_language_models = ["ada","babbage","curie","davinci"]
	valid_whisper_models = ["tiny","base","small","medium","large"]

	def __init__(self):
		# parse arguments:
		parser = argparse.ArgumentParser(description='Voice chat interface for OpenAI')
		parser.add_argument("-rp", "--resume-prompt", help = "Text file from where to resume prompts")
		parser.add_argument("-po", "--prompt-output", help = "Text file for storing prompts to")
		parser.add_argument("-lm", "--language-model", help = "Choose from: "+str(self.valid_language_models))
		parser.add_argument("-wm", "--whisper-model", help = "Choose from: "+str(self.valid_whisper_models))
		parser.add_argument("-mt", "--max-tokens", help = "Maximum amount of tokens")
		parser.add_argument("-lt", "--local-transcription", help = "Set for using local transcription", action='store_true')
		args = parser.parse_args()

		# are we even using whisper?
		if not args.local_transcription:
			self.local_transcription = False
		# what whisper model are we using if any?
		elif args.whisper_model in self.valid_whisper_models:
			whisper_model=args.whisper_model
		else:
			print("Whisper model must be one of the following: "+str(self.valid_whisper_models)+" !!")
			print("Using default: small")
			whisper_model="small"

		# do we store stuff?
		self.prompt_output = args.prompt_output

		# are we resuming a past conversation?
		if args.resume_prompt is None:
			resume_prompt = "init_prompt.txt" 
		else:
			resume_prompt = args.resume_prompt

		with open(resume_prompt, "r") as f:
			self.prompt = f.read()

		# what language model to use?
		if args.language_model in self.valid_language_models:
			self.engine=args.language_model
		else:
			print("Language model must be one of the following: "+str(self.valid_language_models)+" !!")
			print("Using default: babbage")
			self.engine="babbage"

		# audio stuff
		self.q = queue.Queue()
		self.device = 'default'
		device_info = sd.query_devices(self.device, 'input')
		self.samplerate = int(device_info['default_samplerate'])
		self.recordingfile = tempfile.mktemp(prefix='/tmp/tmp_recording', suffix='.wav', dir='')
		self.speechfile = tempfile.mktemp(prefix='/tmp/tmp_speech', suffix='.wav', dir='')
		self.channels = 1
		self.subtype = "PCM_24"

		# GUI stuff
		self.tk = tk.Tk()
		self.button_rec = Button(self.tk, text='Speak', command=self.record)
		self.button_rec.pack()

		self.button_stop = Button(self.tk, text='Done speaking', command=self.stop)
		self.button_stop.pack()

		self.button_cancel = Button(self.tk, text='Cancel recording', command=self.cancel_recording)
		self.button_cancel.pack()

		# Speech recognition
		if self.local_transcription:
			self.model = whisper.load_model(whisper_model)

		# Speech to text
		self.language = 'en'

		# GPT3:
		openai.api_key = os.getenv("OPENAI_API_KEY")
		# how long should a string be?
		if args.max_tokens is None:
			self.max_tokens = 1024
		else:
			self.max_tokens = int(args.max_tokens)

	# Record audio
	def recording_thread(self):
		with sf.SoundFile(self.recordingfile, mode='x', samplerate=self.samplerate, channels=self.channels, subtype=self.subtype) as file:
			with sd.InputStream(samplerate=self.samplerate, device=self.device, channels=self.channels, callback=self.callback):
				while self.running:
					file.write(self.q.get())

	def cancel_recording(self):
		if self.running:
			self.running = False
			self.recordingThread.join()
			os.remove(self.recordingfile)
			print("Recording stopped")
		else:
			print("Already stopped")

	def stop(self):
		if self.running:
			self.running = False
			self.recordingThread.join()
			text = self.speech_to_text()
			text = self.submit_chat(text)
			if len(text) > 0:
				self.speak(text)
			os.remove(self.recordingfile)
			print("Recording stopped")
			if self.prompt_output is not None:
				print("Backing up log to "+self.prompt_output)
				with open(self.prompt_output, "w") as f:
					f.write(self.prompt)
					f.close()
		else:
			print("Already stopped")

	def record(self):
		if self.running:
			print("Already running")
		else:
			self.running = True
			self.recordingThread = Thread(target=self.recording_thread)
			self.recordingThread.start()
			print("Started recording")

	def callback(self, indata, frames, time, status):
		self.q.put(indata.copy())

	# Speech to text
	def speech_to_text(self):
		if self.local_transcription:
			audio = whisper.load_audio(self.recordingfile)
			audio = whisper.pad_or_trim(audio)
			mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
			options = whisper.DecodingOptions(fp16 = False)
			result = whisper.decode(self.model, mel, options)
			return result.text
		else:
			'''
			curl 
			-X POST https://api.openai.com/v1/engines/audio-transcribe-001/transcriptions
			-H "Content-Type: multipart/form-data"
			-H "accept: application/json"
			-s -F 'file=@stackoverflow.webm;type=video/webm'
			-H "Authorization: Bearer $OPENAI_API_KEY"
			'''
			webm_file = self.recordingfile.replace('.wav','.webm')
			webm = ffmpeg.input(self.recordingfile).output(webm_file).overwrite_output()
			webm.run_async(pipe_stdout=False, quiet=True)
			#webm.run(quiet=True)

			buffer = BytesIO()
			c = pycurl.Curl()
			c.setopt(c.WRITEDATA, buffer)
			c.setopt(c.URL, "https://api.openai.com/v1/engines/audio-transcribe-001/transcriptions")
			c.setopt(c.HTTPHEADER, [
				"Content-Type: multipart/form-data",
				"Accept: application/json",
				"Authorization: Bearer "+openai.api_key
			])
			c.setopt(c.HTTPPOST, [
				('file', (
					c.FORM_FILE, webm_file,
					c.FORM_FILENAME, webm_file,
					c.FORM_CONTENTTYPE, 'video/webm',
			   )),
			])
			c.perform()
			c.close()

			dictionary = json.loads(buffer.getvalue())
			ret = dictionary['text']
			os.remove(webm_file)
			return ret 
	
	# Text to speech
	def speak(self, text):
		myobj = gTTS(text=text, lang=self.language, slow=False)
		myobj.save(self.speechfile)
		os.system("mplayer "+self.speechfile)
		os.remove(self.speechfile)
	
	# Send to GPT3
	def submit_chat(self, text):
		template = "Human: {}\nAI:"
		self.prompt += template.format(text)
		result = openai.Completion.create(
			engine=self.engine,
			prompt=self.prompt,
			temperature=0.9,
			max_tokens=self.max_tokens,
			stop=["Human", "AI", "\n"],
			top_p=1.0,
			frequency_penalty=0.0,
			presence_penalty=-0.6
		)
		ret = result.choices[0]['text']
		self.prompt += ret
		self.prompt += '\n' 
		return ret

	def run(self):
		self.tk.mainloop()

main = GPTVoiceInterface()
main.run()
