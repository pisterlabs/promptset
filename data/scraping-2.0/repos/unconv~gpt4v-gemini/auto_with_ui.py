from multiprocessing import Process, Queue, Value
import modules.cv2_stream as cv2_stream
from playsound import playsound
from openai import OpenAI
from queue import Empty
from PIL import Image
import numpy as np
import textwrap
import base64
import time
import sys
import cv2
import os

import modules.recorder as recorder
import modules.helpers as helpers

client = OpenAI()

def motion_fn(queue: Queue, stream_url, processing: Value, frame_queue: Queue):
    for collage in cv2_stream.detect_changes(stream_url, processing=processing, frame_queue=frame_queue):
        print("Motion detected!")
        collage.save("collage.jpg", format="JPEG")

        queue.put({
            "image": collage
        })

        with processing.get_lock():
            processing.value = False

def voice_fn(queue: Queue, processing: Value, ui_queue: Queue, winwidth, winheight):
    for audio_file in recorder.live_speech(60, transcribe_audio=False, processing=processing, ui_queue=ui_queue, winwidth=winwidth, winheight=winheight):
        if os.path.exists("collage.jpg"):
            image = Image.open("collage.jpg")
            os.remove("collage.jpg")
        else:
            frame_rgb = cv2.cvtColor(frame_queue.get(), cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            #image.save("still.jpg", format="JPEG")

        queue.put({
            "audio": audio_file,
            "image": image
        })

        with processing.get_lock():
            processing.value = False

def parse_message(message):
    image = helpers.image_b64(message["image"])

    with open("capture.jpg", "wb") as f:
        captured_image = base64.b64decode(image)
        f.write(captured_image)

    content = [
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image}"
        }
    ]

    if "audio" in message:
        text = recorder.transcribe(message["audio"])

        if helpers.filter_garbage(text):
            return []

        content.append({
            "type": "text",
            "text": text,
        })
    else:
        if os.path.exists("collage.jpg"):
            os.remove("collage.jpg")

    return content

def empty_queue(queue):
    try:
        while True:
            queue.get_nowait()
    except Empty:
        pass

def draw_text(window, text, position, color=(255, 255, 255), centered=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    lines = textwrap.wrap(text, width=50)

    line_y = 0
    for line in lines:
        pos = position

        if centered:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = position[0] - text_size[0] // 2
            text_y = position[1] - text_size[1] // 2
            pos = (text_x, text_y)

        pos = (pos[0], pos[1]+line_y)
        line_y += int(text_size[1]*1.3)

        cv2.putText(window, line, pos, font, font_scale, color, thickness)

    cv2.imshow('GPT4GEMINI', window)
    if cv2.getWindowProperty('GPT4GEMINI', cv2.WND_PROP_VISIBLE) < 1:
        sys.exit()
    cv2.waitKey(1)

def draw_window(winwidth, winheight, queue: Queue, frame_queue: Queue):
    cv2.namedWindow('GPT4GEMINI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('GPT4GEMINI', winwidth, winheight)

    window = np.zeros((winheight, winwidth, 3), dtype=np.uint8)
    cv2.imshow('GPT4GEMINI', window)
    if cv2.getWindowProperty('GPT4GEMINI', cv2.WND_PROP_VISIBLE) < 1:
        sys.exit()

    while True:
        frame = frame_queue.get()
        try:
            event = queue.get_nowait()
        except Empty:
            frame = cv2.resize(frame, (640, 400))

            x = 50
            y = int(winheight / 2 - frame.shape[0] / 2)

            window[y:frame.shape[0]+y, x:frame.shape[1]+x] = frame
            cv2.imshow('GPT4GEMINI', window)
            cv2.waitKey(1)
            continue

        if event["type"] == "draw_text":
            draw_text(window, **event["args"])

        if event["type"] == "clear":
            window = np.zeros((winheight, winwidth, 3), dtype=np.uint8)
            cv2.imshow('GPT4GEMINI', window)

        cv2.waitKey(1)

    cv2.destroyAllWindows()

stream_url = helpers.get_stream()

if os.path.exists("collage.jpg"):
    os.remove("collage.jpg")

if os.path.exists("detect.jpg"):
    os.remove("detect.jpg")

#if os.path.exists("frame.jpg"):
    #os.remove("frame.jpg")

winwidth = int(1920*0.8)
winheight = int(1080*0.8)

user_text_color = (255, 255, 0)

queue = Queue()
ui_queue = Queue()
frame_queue = Queue()

processing_audio = Value('b', False)
processing_video = Value('b', False)

print("Starting motion process...")
motion_process = Process(target=motion_fn, args=(queue, stream_url, processing_video, frame_queue))
motion_process.start()

print("Starting voice process...")
voice_process = Process(target=voice_fn, args=(queue, processing_audio, ui_queue, winwidth, winheight))
voice_process.start()

print("Starting UI process...")
ui_process = Process(target=draw_window, args=(winwidth, winheight, ui_queue, frame_queue))
ui_process.start()

messages = [
    {
        "role": "system",
        "content": """You are an AI assistant that reports what it sees in short sentences. You are connected to a video feed and will be shown a collage of frames from a video and you will tell what what is happening in the video as if you are seeing it live. Don't let the user know that you are seeing a sequence of images or frames. Don't say anything about a series of images or frames. Answer as if you are seeing the sequence of images in real life. Don't say they are a sequence of images. Just say what is happening in them, as if it is happening in front of your eyes. If the user asks you a direct question, answer it based on what you see. You are talking directly to the user. Assume any activity is performed by the user, who you shall refer to as "You". Example responses: "I see you drew an elephant" or "I see a cat" or "I see you put a coin on the table". If you notice something out of the ordinary, point it out. Keep your answers very concise. When playing games, tell the user if they won / are correct. If nothing particularly interesting happened since the previous image sequence, answer with the JSON {"status": "NO_CHANGE"}""".strip(),
    }
]

while True:
    message = queue.get()

    content = parse_message(message)

    have_text = False

    for msg in content:
        if "text" in msg:
            have_text = True
            ui_queue.put({
                "type": "clear",
            })
            ui_queue.put({
                "type": "draw_text",
                "args": {
                    "text": msg["text"],
                    "position": (winwidth//2, 80),
                    "color": user_text_color
                }
            })
            print("You: " + msg["text"])

    time.sleep(0.5)

    if processing_audio.value == True:
        while processing_audio.value == True:
            time.sleep(0.1)

    if processing_video.value == True:
        while processing_video.value == True:
            time.sleep(0.1)

    with processing_audio.get_lock():
        processing_audio.value = True

    try:
        message2 = queue.get(timeout=0.5)
        if message2:
            msg2 = parse_message(message2)
            for msg in msg2:
                if "text" in msg:
                    have_text = True
                    ui_queue.put({
                        "type": "clear",
                    })
                    ui_queue.put({
                        "type": "draw_text",
                        "args": {
                            "text": msg["text"],
                            "position": (winwidth//2, 80),
                            "color": user_text_color
                        }
                    })
                    print("You: " + msg["text"])
            content += msg2
    except Empty:
        pass

    if len(content) == 0:
        continue

    messages.append({
        "role": "user",
        "content": content
    })

    if True:
        print("Sending GPT4V request...")

        # show "calling gpt4v..."
        ui_queue.put({
            "type": "draw_text",
            "args": {
                "text": "calling gpt4v...",
                "position": (winwidth//2, int(winheight*0.9)),
                "color": (255, 0, 255)
            }
        })

        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4-vision-preview",
            max_tokens=1024
        )

        response_message = response.choices[0].message
        response_text = response_message.content

        if '{"status": "NO_CHANGE"}' in response_text:
            messages.pop()
            continue
    else:
        response_text = "This is a test"
        response_message = {
            "role": "assistant",
            "content": response_text,
        }

    messages.append(response_message)

    print("Generating audio response...")

    # hide "calling gpt4v..."
    ui_queue.put({
        "type": "draw_text",
        "args": {
            "text": "calling gpt4v...",
            "position": (winwidth//2, int(winheight*0.9)),
            "color": (0, 0, 0)
        }
    })

    # show "generating audio..."
    ui_queue.put({
        "type": "draw_text",
        "args": {
            "text": "generating audio...",
            "position": (winwidth//2, int(winheight*0.9)),
            "color": (0, 255, 255)
        }
    })

    audio = client.audio.speech.create(
        input=response_text,
        model="tts-1",
        voice="onyx",
    )

    audio.stream_to_file("audio.mp3")

    # hide "generating audio..."
    ui_queue.put({
        "type": "draw_text",
        "args": {
            "text": "generating audio...",
            "position": (winwidth//2, int(winheight*0.9)),
            "color": (0, 0, 0)
        }
    })

    if not have_text:
        ui_queue.put({
            "type": "clear",
        })

    ui_queue.put({
        "type": "draw_text",
        "args": {
            "text": response_text,
            "position": (int(winwidth*0.71), int(winheight*0.5))
        }
    })
    print("GPT: " + response_text)

    playsound("audio.mp3")
    os.remove("audio.mp3")

    time.sleep(1)

    with processing_audio.get_lock():
        processing_audio.value = False

    with processing_video.get_lock():
        processing_video.value = False

    empty_queue(queue)

video_process.join()
motion_process.join()
voice_process.join()
ui_process.join()
