# Study-Bot: Question answering using audio interaction and object detection, 
# CLI version for testing, and functions for GUI version

import openai
import whisper
from pydub import AudioSegment
from pydub.playback import play as pydubPlay
import io
from typing import Iterator
import pyaudio
import wave
from pathlib import Path
# History, is_installed, and subprocess are used only by main script, do not remove
from elevenlabs import set_api_key, generate, History, is_installed, subprocess
import cv2
import numpy as np
import time
import threading
import keyboard
import credentials # Ignored by git, contains API keys
import sourceMaterial

global objects
global question
global answer
global topic
global stop
objects = ''
question = ''
answer = ''
stop = False

GPT_MODEL = 'gpt-3.5-turbo-16k'

# Credentials
openai.api_key = credentials.openAIKey
set_api_key(credentials.elevenLabsKey)

# Behavioral guidelines for conversation
instructions = """
Try to use the information below to help the user study by answering 
the user's question. The user may or may not be holding a physical representation 
of what their question is about. Consider the object list, which includes all 
the objects that the user is holding, so that the answer can be refined to be 
more specific to the user's question. Do not mention the user or the information 
in your answer to make it more sound natural.

If the question is unrelated to the information, ignore all previous instructions
and try to answer the question without mentioning the information or the objects 
to make it sound more natural.

Always try to give brief answers to the user's questions.
"""

# Recorder configuration
CHUNK = 1024 # Chunk size
FORMAT = pyaudio.paInt16 # Audio codec format
CHANNELS = 2
RATE = 44100 # Sample rate
OUTPUT_FILE = 'question.wav'

def recordQuestion():
	global question
	global stop

	stop = False
	audio = pyaudio.PyAudio() # Initialize PyAudio
	# Open audio stream for recording
	stream = audio.open(format = FORMAT, channels = CHANNELS, rate = RATE, input = True, frames_per_buffer = CHUNK)
	frames = []

	# Record audio stream in chunks
	while not stop:
		data = stream.read(CHUNK)
		frames.append(data)

	# Stop and close audio stream
	stream.stop_stream()
	stream.close()
	audio.terminate()

	# Save recording as WAV
	wf = wave.open(OUTPUT_FILE, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(audio.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

	# STT Conversion
	model = whisper.load_model('base')
	result = model.transcribe(OUTPUT_FILE, fp16 = False)
	question = result['text']
	
	# Delete audio file
	Path(OUTPUT_FILE).unlink()

def stopRecording():
	global stop
	stop = True

def colorID():
	obj = 'User is not holding any objects'

	# Capture video
	cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Use 0 for default camera

	# Start timer
	startTime = time.time()
	elapsedTime = 0

	# Color ranges
	stomachLower = np.array([90, 	80, 		100			], np.uint8)
	stomachUpper = np.array([120, 	255, 		255			], np.uint8)
	colonLower = np.array(	[10, 	255 * 0.55, 255 * 0.35	], np.uint8)
	colonUpper = np.array(	[19.5, 	255, 		255			], np.uint8)
	liverLower = np.array(	[38, 	225 * 0.22, 255 * 0.38	], np.uint8)
	liverUpper = np.array(	[41, 	255, 		255			], np.uint8)
	brainLower = np.array(	[161, 	255 * 0.50, 255 * 0.40	], np.uint8)
	brainUpper = np.array(	[161, 	255, 		255			], np.uint8)
	kidneyLower = np.array(	[26, 	255 * 0.60, 255 * 0.69	], np.uint8)
	kidneyUpper = np.array(	[26, 	255, 		255			], np.uint8)
	heartLower = np.array(	[179, 	255 * 0.50, 255 * 0.35	], np.uint8)
	heartUpper = np.array(	[179, 	255 * 0.97, 255 * 0.69	], np.uint8)

	while elapsedTime < 1:

		_, imageFrame = cam.read()

		# Convert frame from BGR color space to HSV
		hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

		# Create masks for each organ
		colonMask = cv2.inRange(hsvFrame, colonLower, colonUpper)
		liverMask = cv2.inRange(hsvFrame, liverLower, liverUpper)
		stomachMask = cv2.inRange(hsvFrame, stomachLower, stomachUpper)
		brainMask = cv2.inRange(hsvFrame, brainLower, brainUpper)
		kidneyMask = cv2.inRange(hsvFrame, kidneyLower, kidneyUpper)
		heartMask = cv2.inRange(hsvFrame, heartLower, heartUpper)

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

		# For brain
		brainMask = cv2.dilate(brainMask, kernel)
		resBrain = cv2.bitwise_and(imageFrame, imageFrame, mask=brainMask)

		# For heart
		heartMask = cv2.dilate(heartMask, kernel)
		resHeart = cv2.bitwise_and(imageFrame, imageFrame, mask=heartMask)

		# For kidney use a more aggressive kernel for dilation
		kidneyMask = cv2.dilate(kidneyMask, np.ones((12, 12), 'uint8'))
		resKidney = cv2.bitwise_and(imageFrame, imageFrame, mask=kidneyMask)

		# Create a contour around the zone that matches the color range
		contours, hierarchy = cv2.findContours(colonMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# For each countour, check if the area is greater than the threshold
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 700:
				# Append the name of the model to the list of objects
				if 'colon' not in obj:
					if obj == 'User is not holding any objects':
						obj = 'colon'
					else:
						obj = obj + ', colon'

		contours, hierarchy = cv2.findContours(liverMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 500:
				if 'liver' not in obj:
					if obj == 'User is not holding any objects':
						obj = 'liver'
					else:
						obj = obj + ', liver'

		contours, hierarchy = cv2.findContours(stomachMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 1400:
				if 'stomach' not in obj:
					if obj == 'User is not holding any objects':
						obj = 'stomach'
					else:
						obj = obj + ', stomach'

		contours, hierarchy = cv2.findContours(brainMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 2500:
				if 'brain' not in obj:
					if obj == 'User is not holding any objects':
						obj = 'brain'
					else:
						obj = obj + ', brain'
		
		contours, hierarchy = cv2.findContours(heartMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 650:
				if 'heart' not in obj:
					if obj == 'User is not holding any objects':
						obj = 'heart'
					else:
						obj = obj + ', heart'

		contours, hierarchy = cv2.findContours(kidneyMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > 50:
				if 'kidney' not in obj:
					if obj == 'User is not holding any objects':
						obj = 'kidney'
					else:
						obj = obj + ', kidney'

		elapsedTime = time.time() - startTime

		# This does not break the loop, but removing it breaks the camera feed and causes the program to crash
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release webcam and close all windows
	cam.release()
	cv2.destroyAllWindows()

	return obj

def markerID():
	obj = 'User is not holding any objects'

	# Choose the predefined dictionary to use
	arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

	# Define the names of the objects
	compoundDict = { 0: 'Citrate', 1: 'Isocitrate', 2: 'Alpha-Ketoglutarate', 3: 'Succinyl CoA', 4: 'Succinate', 5: 'Fumarate', 6: 'Malate', 7: 'Oxaloacetate' }

	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Use 0 for default camera

	# Start timer
	startTime = time.time()
	elapsedTime = 0


	while elapsedTime < 5:
		ret, frame = cap.read()

		if not ret:
			print('Failed to capture frame.')
			break

		# Convert the frame to grayscale for marker detection
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect markers
		corners, ids, _ = cv2.aruco.detectMarkers(gray, arucoDict)

		if ids is not None:
			# For each marker found in current frame
			for i in range(len(ids)):
				try:
					# Try to get the name of the compound from the dictionary
					compoundName = compoundDict[ids[i][0]]
					# Append compound to list while avoiding repeats
					if obj == 'User is not holding any objects':
						obj = compoundName
					elif compoundName not in obj:
						obj += ', ' + compoundName
				except KeyError:
					continue

		# Display the frame
		cv2.imshow('Study-Bot View', frame)

		elapsedTime = time.time() - startTime

		# Check for 'Esc' key press
		key = cv2.waitKey(10) & 0xFF
		if key == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

	return obj

def lookForObjects(topic: int):
	global objects
	objects = ''

	if topic == 1:
		# Call the function for color identification
		objects = colorID()
	elif topic == 2:
		# Call the function for marker identification
		objects = markerID()

def sendMessage(messageList: any):
	# Send prompt to GPT
	response = openai.ChatCompletion.create(
		messages = messageList,
		model = GPT_MODEL, 
		temperature = 0.2
	)
	# print(response)
	_answer = response['choices'][0]['message']['content']

	# Add the response to the message list
	messageList.append({'role': 'assistant', 'content': _answer})

def streamAnswer(audioStream: Iterator[bytes]) -> bytes:
	audioOutput = b''
	
	# For each chunk of audio in stream, add it to the output
	for chunk in audioStream:
		if chunk is not None:
			audioOutput += chunk

	# Play audio output using PyDub
	audioSegment = AudioSegment.from_file(io.BytesIO(audioOutput), format="mp3")
	pydubPlay(audioSegment)

def convertTTS(text: str):
	audioOutput = generate(text = text, model = 'eleven_multilingual_v1', stream = True)
	streamAnswer(audioOutput)
	# print('Audio playback disabled.\n')

# Only run if not imported as a module
if __name__ == '__main__':
	# Listen for keyboard input to stop recording
	keyboard.add_hotkey('s', stopRecording)

	print('Select a topic NUMBER from the list:\n')
	print('[1] - Human Body')
	print('[2] - Krebs Cycle\n')
	topic = int(input('Topic: '))
	source = ''

	# Load the source material based on the selected topic
	if topic == 1:
		print('Topic: Human Body\n')
		source = sourceMaterial.humanBody
	elif topic == 2:
		print('Topic: Krebs Cycle\n')
		source = sourceMaterial.krebsCycle

	# Start question processing threads
	objID = threading.Thread(target = lookForObjects, args = (topic,))
	audioRec = threading.Thread(target = recordQuestion)

	objID.start()
	print('Looking for objects...\n')
	audioRec.start()
	print('Listening for question...\n')

	objID.join()
	print('Object detection complete.\n')
	print('Objects detected: ' + objects + '\n')
	audioRec.join()
	print('Question recorded.\n')
	print('Question: ' + question + '\n')

	# Build prompt
	query = f"""{instructions}

	Objects held by user: {objects}.
	Question: {question}

	Information: 
	\"\"\"
	{source}
	\"\"\"
	"""

	# Send prompt to GPT
	messageHistory = [
		{'role': 'system', 'content': 'You answer questions in the same language as the question.'},
		{'role': 'user', 'content': query},
	]

	print('Sending prompt to GPT...\n')
	sendMessage(messageHistory)
	# Get the answer from the last message in the message history
	answer = next((msg for msg in reversed(messageHistory) if msg['role'] == 'assistant'), None)['content']
	print(answer + '\n')
	
	if answer != '':
		print('Answer: ' + answer + '\n\n')

	# Convert answer to audio
	print('Converting answer to audio...\n')
	convertTTS(answer)

	# Conversation loop, handles any follow-up questions
	while True:
		print('Press space to ask another question, or press q to quit.\n')

		while True:
			if keyboard.is_pressed(' '):
				print('Preparing for next question, please hold...\n')
				break
			if keyboard.is_pressed('q'):
				print('Exiting program...\n')
				exit()

		# Reset variables
		objects = 'User is not holding any objects'
		question = ''

		# Restart threads

		objID = threading.Thread(target = lookForObjects, args = (topic,))
		audioRec = threading.Thread(target = recordQuestion)

		objID.start()
		print('Looking for objects...\n')
		audioRec.start()
		print('Listening for question...\n')

		objID.join()
		print('Object detection complete.\n')
		print('Objects detected: ' + objects + '\n')
		audioRec.join()
		print('Question recorded.\n')
		print('Question: ' + question + '\n')


		# NOTE: Add this to the query if the model's response has any deviations from previous instructions
		"""
		Remember to consider the object list and the information provided when answering 
		the user's question. Do not mention the user or the information in your answer 
		to make it more sound natural.

		If the question is unrelated to the information, ignore all previous instructions
		and try to answer the question without mentioning the information or the objects
		to make it sound more natural.
		"""

		# Build new prompt and add to chat history
		query = f"""Objects held by user: {objects}.
Question: {question}
"""
		messageHistory.append({'role': 'user', 'content': query})
		answer = ''

		# Send prompt to GPT
		# print('Prompt: ' + query + '\n') # For debugging only
		print('Sending prompt to GPT...\n')

		sendMessage(messageHistory)
		answer = next((msg for msg in reversed(messageHistory) if msg['role'] == 'assistant'), None)['content']

		if answer != '':
			print('Answer: ' + answer + '\n\n')

		print('Converting answer to audio...\n')
		convertTTS(answer)