from google.cloud import vision
import io
from google.cloud.vision import types
from DenseCapCaptioning import get_captions
from geolocation import get_location, get_weather
from datetime import date



import os 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './secret.json'
os.system("pip install openai")  
# Instantiates google client 
client = vision.ImageAnnotatorClient()
import openai
import cv2
from imutils.video import VideoStream
from flask import Flask, Response, render_template, jsonify
outputFrame = None
camera = cv2.VideoCapture(0)
import threading
lock = threading.Lock()


app = Flask(__name__)

# initialize the video stream
vs = VideoStream(src=0).start()

# warmup
import time
time.sleep(2.0)
openai.api_key = "" 
with open('./prompts/gpt3prompts.txt') as file:
	danabot = file.read()
def get_reply(msg, training=danabot, temperature = 0.2):
	prompt = training+"[[Message]]:"+msg+"\n[[Response]]:"
	response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=30, stop=['\n\n'])
	reply = response["choices"][0]["text"]  
	return reply 	

def image_to_path(frame):
	flag, encodedImage = cv2.imencode(".jpg", frame)
	file_name = 'image.jpg'
	returnvalue, encodedImage = camera.read()
	cv2.imwrite(file_name, encodedImage)
	#with io.open(file_name, 'rb') as image_file:
		#content = image_file.read()
	return  './'+file_name

def image_caption(frame):
	return "a dog walked around the person"

def ocr_detection(frame):
	global client
	content = image_to_path(frame)
	image = vision.types.Image(content=content)
	response = client.text_detection(image=image)
	texts = response.text_annotations
	ocrs = [] 
	for text in texts:
		ocrs.append(text.description) 
        #print('\n"{}"'.format(text.description))
		vertices = (['({},{})'.format(vertex.x, vertex.y)
					for vertex in text.bounding_poly.vertices])
        #print('bounds: {}'.format(','.join(vertices)))
	# instead of bounds top left top right have thresholds 
	return ocrs 
   

def label_detection(frame):

	global client

	content = image_to_path(frame)

	image = types.Image(content=content)

	# Performs label detection on the image file
	response = client.label_detection(image=image)
	labels = response.label_annotations
	return labels

def add_timestamp(frame):

	# grab the current timestamp and draw it on the frame
	from datetime import datetime
	timestamp = datetime.now()
	cv2.putText(frame, timestamp.strftime(
		"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

def process_video():
	
	global vs, outputFrame, lock

	# loop over frames from the video stream
	while True:

		# read the next frame from the video stream
		frame = vs.read()

		# various detections
		add_timestamp(frame)

		# acquire the lock, set the output frame, and release the lock
		with lock:
			outputFrame = frame.copy()

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/text_feed") 
def text_feed(): 
	def generate(): 
		global outputFrame, lock

		# loop over frames from the output stream
		while True:
			print(get_captions(image_to_path(outputFrame)))

			# wait until the lock is acquired
			with lock:

				# check if the output frame is available, otherwise skip the iteration of the loop
				if outputFrame is None: continue
				today = date.today()
				d1 = today.strftime("%d/%m/%Y")

				prescene = { 
					"timestamp": add_timestamp(outputFrame),
					"caption": get_captions(image_to_path(outputFrame)),
					"weather": get_weather(),
					"location": get_location(),
					#"s": label_detection(outputFrame),
					#"ocr": ocr_detection(outputFrame)
				} 
				gpt3caption= get_reply(prescene["caption"] + "| " + 
						 prescene["weather"] + "| " + 
						 prescene["location"]  + "| " +
						 d1)

				scene = { 
					"caption": gpt3caption,
					#"labels": label_detection(outputFrame),
					#"ocr": ocr_detection(outputFrame)
				} 
				scene= [gpt3caption]
			
			yield f"data: {scene}\n\n"
		today = date.today()
		d1 = today.strftime("%d/%m/%Y")
		 


	return Response(generate(), mimetype = "text/event-stream")

@app.route("/video_feed")
def video_feed():
	returnvalue, output = camera.read()
	def generate():
		# loop over frames from the output stream
		while True:

			# wait until the lock is acquired

			# encode the frame in JPEG format
			flag, encodedImage = cv2.imencode(".jpg", output)

			# ensure the frame was successfully encoded
			if not flag: continue

			# yield the output frame in the byte format
			yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'

	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	
	t = threading.Thread(target=process_video)
	t.daemon = True
	t.start()

	app.run(host="0.0.0.0", port="8000", debug=True, use_reloader=False)

# release the video stream pointer
vs.stop()
