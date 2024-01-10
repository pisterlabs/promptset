# Script for testing multithreading by listening for question and looking for objects at the same time

import threading
import openai
import whisper
import pyaudio
import wave
import ffmpeg
from pathlib import Path
import cv2
import numpy as np
import time

openai.api_key = ''

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"


def look():
		# Capture video
	cam = cv2.VideoCapture(0) # Use 0 for default camera
	print('Looking for objects...\n')
	
	objects = 'User is not holding any objects'

	startTime = time.time()
	elapsedTime = 0

	# Stomach color range
	stomachLower = np.array([90, 80, 1], np.uint8)
	stomachUpper = np.array([120, 255, 255], np.uint8)

	# Colon color range
	colonLower = np.array([9, 255 * 0.55, 255 * 0.35], np.uint8)
	colonUpper = np.array([28, 255, 255], np.uint8)

	# Liver color range
	liverLower = np.array([38, 225 * 0.22, 255 * 0.38], np.uint8)
	liverUpper = np.array([41, 255, 255], np.uint8)

	while elapsedTime < 2:

		_, imageFrame = cam.read()

		# Convert frame from BGR color space to HSV
		hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

		# Create masks for each organ
		colonMask = cv2.inRange(hsvFrame, colonLower, colonUpper)
		liverMask = cv2.inRange(hsvFrame, liverLower, liverUpper)
		stomachMask = cv2.inRange(hsvFrame, stomachLower, stomachUpper)

		# Create a 5x5 square-shaped filter called kernel
		# Filter is filled with ones and will be used for morphological transformations such as dilation for better detection
		kernel = np.ones((5, 5), 'uint8')


		# For colon
		# Dilate mask: Remove holes in the mask by adding pixels to the boundaires of the objects in the mask
		colonMask = cv2.dilate(colonMask, kernel)
		# Apply mask to frame by using bitwise AND operation
		resColon = cv2.bitwise_and(imageFrame, imageFrame, mask = colonMask)


		# For liver
		liverMask = cv2.dilate(liverMask, kernel)
		resliver = cv2.bitwise_and(imageFrame, imageFrame, mask=liverMask)

		# For stomach
		stomachMask = cv2.dilate(stomachMask, kernel)
		resStomach = cv2.bitwise_and(imageFrame, imageFrame, mask=stomachMask)

		# Create a contour around the zone that matches the color range
		contours, hierarchy = cv2.findContours(colonMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# For each countour, check if the area is greater than the threshold
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 500:
				x, y, w, h = cv2.boundingRect(contour)
				imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 120, 255), 2)
				cv2.putText(imageFrame, "COLON", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 120, 255))
		        # Append the name of the model to the list of objects
				if 'colon' not in objects:
					if objects == 'User is not holding any objects':
						objects = 'colon'
					else:
						objects = objects + ', colon'

		contours, hierarchy = cv2.findContours(liverMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 500:
				x, y, w, h = cv2.boundingRect(contour)
				imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (86, 194, 0), 2)
				cv2.putText(imageFrame, "LIVER", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (86, 194, 0))
				if 'liver' not in objects:
					if objects == 'User is not holding any objects':
						objects = 'liver'
					else:
						objects = objects + ', liver'

		contours, hierarchy = cv2.findContours(stomachMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 1400:
				x, y, w, h = cv2.boundingRect(contour)
				imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (237, 117, 47), 2)
				cv2.putText(imageFrame, "STOMACH", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (237, 117, 47))
				if 'stomach' not in objects:
					if objects == 'User is not holding any objects':
						objects = 'stomach'
					else:
						objects = objects + ', stomach'

		# Display the camera feed
		cv2.imshow('Study-Bot View', imageFrame)

		elapsedTime = time.time() - startTime

		# This does not stop the program from running, but removing it breaks the camera feed and causes the program to crash
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release webcam and close all windows
	cam.release()
	cv2.destroyAllWindows()

	print('Camera closed\n')
	print('Objects detected: ' + objects + '\n')

def listen():
	audio = pyaudio.PyAudio()
	stream = audio.open(format = FORMAT, channels = CHANNELS, rate = RATE, input = True, frames_per_buffer = CHUNK)

	print('Listening for question...\n')

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)

	stream.stop_stream()
	stream.close()
	audio.terminate()

	print('Recording stopped.\n')
	print('Saving and converting audio...\n')
	print('------------------------------\n')

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(audio.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

	inputAudio = ffmpeg.input('output.wav')
	outputAudio = ffmpeg.output(inputAudio, 'output.mp3')
	ffmpeg.run(outputAudio)

	print('\n------------------------------\n')
	print('Audio saved as: output.mp3')

t1 = threading.Thread(target = listen)
t2 = threading.Thread(target = look)

t1.start()
t2.start()

t1.join()
t2.join()

print('Converting audio to text...\n')

model = whisper.load_model('base')
result = model.transcribe('output.wav', fp16 = False, language = 'English')
print('Question: ' + result['text'] + '\n')

# Delete audio files
Path('output.wav').unlink()
Path('output.mp3').unlink()

print('-------------- DONE --------------\n')