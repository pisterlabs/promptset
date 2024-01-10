import calculations
import json
import openai
import os
from datetime import datetime
from dotenv import load_dotenv
from subprocess import Popen, PIPE

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def append_to_log(filename, contents):
	with open(filename, "a") as file:
		file.write(contents)

def execute_command(command):
	project_dir = os.path.abspath("./")
	p = Popen(command, cwd=project_dir, stdout=PIPE, stderr=PIPE, shell=True)
	stdout, stderr = p.communicate()
	if stderr.decode("utf-8") != "":
		print(stderr.decode("utf-8"))

def transcribe_audio(timestamp_epoch):
	return openai.Audio.transcribe("whisper-1", open("static/recordings/" + timestamp_epoch + ".mp3", "rb"))["text"]

def update_audio_file(timestamp_readable, timestamp_epoch):
	lines = []
	with open("logs/recordings.json") as file:
		for line in file.readlines():
			lines.append(json.loads(line))
	for line in lines:
		if line["timestamp"] == timestamp_readable:
			line["audio_file"] = "recordings/" + timestamp_epoch + ".mp3"
	with open("logs/recordings.json", "w") as file:
		for line in lines:
			file.write(f"{json.dumps(line)}\n")

def update_transcript(timestamp_readable, transcript):
	lines = []
	with open("logs/recordings.json") as file:
		for line in file.readlines():
			lines.append(json.loads(line))
	for line in lines:
		if line["timestamp"] == timestamp_readable:
			line["transcript"] = transcript
	with open("logs/recordings.json", "w") as file:
		for line in lines:
			file.write(f"{json.dumps(line)}\n")

def start_manual_recording(frequency, seconds_to_record):
	timestamp = datetime.now()
	timestamp_readable = timestamp.strftime("%m-%d-%Y %H:%M:%S")
	timestamp_epoch = timestamp.strftime("%s")
	append_to_log("logs/tracker_output.log", "[" + timestamp_readable + "] Started manual recording on " + frequency + " MHz." + "\n")
	recording_output = {
		"timestamp": timestamp_readable,
		"user_location": "None",
		"iss_location": "None",
		"distance": "None",
		"elevation_angle": "None",
		"frequency": frequency,
		"audio_file": "",
		"transcript": ""
	}
	append_to_log("logs/recordings.json", json.dumps(recording_output) + "\n")
	execute_command("rtl_sdr -f " + frequency + "M -s 256k -n " + str(256000 * int(seconds_to_record)) + " static/recordings/" + timestamp_epoch + ".iq")
	execute_command("cat static/recordings/" + timestamp_epoch + ".iq | ./demodulator.py > static/recordings/" + timestamp_epoch + ".raw")
	execute_command("ffmpeg -f s16le -ac 1 -ar 256000 -acodec pcm_s16le -i static/recordings/" + timestamp_epoch + ".raw -af 'highpass=f=200, lowpass=f=3000, volume=4' static/recordings/" + timestamp_epoch + ".mp3")
	execute_command("rm -rf static/recordings/" + timestamp_epoch + ".iq static/recordings/" + timestamp_epoch + ".raw")
	update_audio_file(timestamp_readable, timestamp_epoch)
	append_to_log("logs/tracker_output.log", "[" + datetime.now().strftime("%m-%d-%Y %H:%M:%S") + "] Saved recording to: " + timestamp_epoch + ".mp3" + "\n")
	transcript = transcribe_audio(timestamp_epoch)
	update_transcript(timestamp_readable, transcript)
	append_to_log("logs/tracker_output.log", "[" + datetime.now().strftime("%m-%d-%Y %H:%M:%S") + "] Finished transcribing audio." + "\n")
