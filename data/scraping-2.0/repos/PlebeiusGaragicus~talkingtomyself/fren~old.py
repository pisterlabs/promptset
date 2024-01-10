import websockets
import asyncio
import base64
import json
import pyaudio
import os
import threading
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).parent



FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


pause_event = threading.Event()

# def listen_for_a_sentence():
# 	p = pyaudio.PyAudio()

# 	# Open an audio stream with above parameter settings
# 	stream = p.open(
# 		format=FORMAT,
# 		channels=CHANNELS,
# 		rate=RATE,
# 		input=True,
# 		frames_per_buffer=FRAMES_PER_BUFFER
# 	)


# 	# Send audio (Input) / Receive transcription (Output)
# 	async def send_receive():
# 		URL = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}"

# 		print(f'Connecting websocket to url ${URL}')

# 		async with websockets.connect(
# 			URL,
# 			extra_headers=(("Authorization", os.getenv("ASSEMBLYAI_API_KEY")),),
# 			ping_interval=5,
# 			ping_timeout=20
# 		) as _ws:

# 			r = await asyncio.sleep(0.1)
# 			print("Receiving messages ...")

# 			session_begins = await _ws.recv()
# 			print(session_begins)
# 			print("Sending messages ...")


# 			async def send():
# 				while True:
# 					try:
# 						if not pause_event.is_set():
# 							data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
# 							if data:  # Check if data is not empty
# 								data = base64.b64encode(data).decode("utf-8")
# 								json_data = json.dumps({"audio_data": str(data)})
# 								print(json_data)
# 								r = await _ws.send(json_data)
# 					except websockets.exceptions.ConnectionClosedError as e:
# 						print(e)
# 						assert e.code == 4008
# 						break
# 					except Exception as e:
# 						print(e)
# 						assert False, "Not a websocket 4008 error"
# 					r = await asyncio.sleep(0.01)


# 			async def receive():
# 				while True:
# 					if pause_event.is_set():
# 						continue

# 					try:
# 						result_str = await _ws.recv()
# 						result = json.loads(result_str)['text']

# 						if json.loads(result_str)['message_type']=='FinalTranscript':
# 							print("Human: ", result)
# 							thread = threading.Thread(target=speak, args=(result,))
# 							thread.start()

# 					except websockets.exceptions.ConnectionClosedError as e:
# 						print(e)
# 						assert e.code == 4008
# 						break

# 					except Exception as e:
# 						print(e)
# 						assert False, "Not a websocket 4008 error"

# 				print("RETURNING FROM RECEIVE")
# 				# return result
				
# 			send_result, receive_result = await asyncio.gather(send(), receive())
# 			return receive_result


# 	asyncio.run(send_receive())

# References (Code modified and adapted from the following)
# 1. https://github.com/misraturp/Real-time-transcription-from-microphone
# 2. https://medium.com/towards-data-science/real-time-speech-recognition-python-assemblyai-13d35eeed226




def listen_for_a_sentence():


	thread = threading.Thread(target=os.system, args=(f"afplay {HERE / 'bling.mp3'}",))
	thread.start()

	p = pyaudio.PyAudio()

    # Open an audio stream
	stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

	transcription_text = asyncio.run(send_receive(stream))
	stream.stop_stream()
	stream.close()
	p.terminate()
	return transcription_text



async def send_receive(stream):
    # print("LISTENING>>>")

    URL = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}"
    async with websockets.connect(
        URL,
        extra_headers=(("Authorization", os.getenv("ASSEMBLYAI_API_KEY")),),
        ping_interval=5,
        ping_timeout=20
    ) as _ws:

        await asyncio.sleep(0.1)  # Small delay for connection stabilization

        # Removed redundant _ws.recv() call - assuming initial handshake is not needed
        await _ws.recv()

        transcription_complete = asyncio.Event()

        async def send():
            while not transcription_complete.is_set():
                try:
                    data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                    if data:  # Check if data is not empty
                        data = base64.b64encode(data).decode("utf-8")
                        json_data = json.dumps({"audio_data": str(data)})
                        await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    print(e)
                    # Log the error but don't assert false to avoid crashing
                await asyncio.sleep(0.01)

        async def receive():
            while not transcription_complete.is_set():
                try:
                    result_str = await _ws.recv()
                    result = json.loads(result_str)['text']

                    if json.loads(result_str)['message_type'] == 'FinalTranscript':
                        print("Human: ", result)
                        transcription_complete.set()

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    print(e)
                    # Log the error but don't assert false to avoid crashing

            print("RETURNING FROM RECEIVE")
            return result

        send_task = asyncio.create_task(send())
        receive_task = asyncio.create_task(receive())

        await asyncio.wait([send_task, receive_task], return_when=asyncio.ALL_COMPLETED)

        return await receive_task






def speak(input: str):
	# Set the pause_event to pause audio recording
	pause_event.set()
	
	print(f"Speaking: {input}")
	
	HERE = Path(__file__).parent

	client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # print(client.models.list())

	speech_file_path = HERE / "speech.mp3"
	response = client.audio.speech.create(
        model="tts-1",
        # voice="alloy",
        voice="onyx",
        input=f"{input}"
    )

	speech_file_path = HERE / "speech.mp3"
	response.stream_to_file(speech_file_path)

    # play the file with afplay
	os.system(f"afplay {speech_file_path}")

	# Clear the pause_event to resume audio recording
	pause_event.clear()
