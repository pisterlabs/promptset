import cv2
import logging
import threading
import queue
from transformers import YolosImageProcessor, YolosForObjectDetection, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from flask import Flask, Response, render_template
from gtts import gTTS
from playsound import playsound
import openai

# Create event to signal threads to stop
stop_signal = threading.Event()

# Web interface
app = Flask(__name__)

# Initialize models and queues
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)
# less precise
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
# more precise
#model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
#image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Frames
frame_queue = queue.Queue(maxsize=2)
latest_frame = None
latest_bg_frame = None
latest_obj_frame = None


def read_frames():
    global stop_signal, latest_frame
    cap = cv2.VideoCapture(0)
    print("Started")
    
    while not stop_signal.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        latest_frame = frame

        try:
            frame_queue.put(frame, timeout=2)
        except queue.Full:
            logging.error("Queue is full")
            pass

    cap.release()


def generate_frames():
    global stop_signal, latest_frame

    while not stop_signal.is_set():
        if latest_frame is not None:
            flag, encoded_image = cv2.imencode('.jpg', latest_bg_frame)
            if flag:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


def process_frame_with_bg_subtraction():
    global stop_signal, latest_bg_frame

    while not stop_signal.is_set():
        try:
            frame = frame_queue.get(timeout=2)
            if frame is None:
                break

            fg_mask = bg_subtractor.apply(frame)
            latest_bg_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

        except queue.Empty:
            pass


def process_frame_with_object_detection():
    global stop_signal, latest_obj_frame

    while not stop_signal.is_set():
        try:
            frame = frame_queue.get(timeout=2)
            if frame is None:
                break

            # Your object detection code
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]

                x1, y1, x2, y2 = map(int, box)

                # Draw rectangle (bounding box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optionally, add text to show confidence score
                cv2.putText(frame, f"{model.config.id2label[label.item()]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Your drawing code
            latest_obj_frame = frame  # Update this frame after drawing bounding boxes

        except queue.Empty:
            pass

def run_web_app():
    global stop_signal

    while not stop_signal.is_set():
        app.run(host='0.0.0.0', port=5000, threaded=True)
        stop_signal.wait(1)  # Check for stop signal every second

def listen_and_reply():
    global stop_signal

    audio_temp_file = 'talk.mp3'
    incoming_vocal_command = ""

    while not stop_signal.is_set():
        if incoming_vocal_command:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=incoming_vocal_command,
                max_tokens=60
            )

            speech = gTTS(text=response.choices[0].text.strip(), lang='en', tld='ie', slow=False)
            speech.save(audio_temp_file)
            playsound(audio_temp_file)

# Start threads
threading.Thread(target=read_frames).start()
threading.Thread(target=process_frame_with_bg_subtraction).start()
threading.Thread(target=process_frame_with_object_detection).start()
threading.Thread(target=run_web_app).start()

# Main display loop
while True:
    if latest_bg_frame is not None:
        cv2.imshow('Background Subtraction', latest_bg_frame)
    if latest_obj_frame is not None:
        cv2.imshow('Object Detection', latest_obj_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        stop_signal.set()
        break

cv2.destroyAllWindows()
