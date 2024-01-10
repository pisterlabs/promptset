import os
import re

import openai
import PyPDF2
from dotenv import load_dotenv
from google.cloud import speech, storage, vision
from pytube import YouTube

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './cred.json'

speech_client = speech.SpeechClient()
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()


def main():
	"""
	print(generate_questions("Perhaps it had something to do with living in a dark cupboard, but Harry had always been small and skinny for his age. "
							"He looked even smaller and skinnier than he really was because all he had to wear were old clothes of Dudley's, and Dudley "
							"was about four times bigger than he was. Harry had a thin face, knobbly knees, black hair, and bright green eyes. He wore "
							"round glasses held together with a lot of Scotch tape because of all the times Dudley had punched him on the nose. The only "
							"thing Harry liked about his own appearance was a very thin scar on his forehead that was shaped like a bolt of lightning. He "
							"had had it as long as he could remember, and the first question he could ever remember asking his Aunt Petunia was how he had gotten it."))
	"""
	#print(generate_questions("Harry was a short boy. Harry had an invisible cape gifted to him at the age of 12. Harry grew up poor and went to Hogwarts."))
	
	
	bucket_name = 'annai_demo_bucket'
	mp3_file_to_upload = './tests/audio_prompt3.mp3'
	cloud_blob_name = 'annai_demo'

	try:
		create_bucket(bucket_name)
	except:
		pass

	# QUESTIONS FROM MP3
	# upload_to_cloud(bucket_name, mp3_file_to_upload, cloud_blob_name)
	# t = transcript_audio(bucket_name, cloud_blob_name)
	# print()
	# print(generate_questions(t))

	# QUESTIONS FROM MP3
	# t2 = transcript_pdf('./tests/pdf_prompt2.pdf')
	# print(generate_questions(t2))
	#youtube_to_mp3("https://www.youtube.com/watch?v=rUWxSEwctFU", './inputs/yt-demo.mp3')
	

# Function to take in text and generate questions
def generate_questions(prompt: str):
	openai_prompt = f"Generate a detailed list of unique reading comprehension questions based on this text:\n\n{prompt}"

	resp = openai.Completion.create(
		model = "text-davinci-002", prompt=openai_prompt, temperature=0.7, max_tokens=256
	)
	questions = resp["choices"][0]["text"].strip()
	questions_arr = re.split('\n', questions)
	questions_arr = [re.sub(r'\d. ', '', q) for q in questions_arr if q and q[-1] == "?"]

	return questions_arr


# ----- GOOGLE CLOUD FUNCTIONS -----
def create_bucket(bucket_name):
	cloud_bucket_name = bucket_name
	bucket = storage_client.bucket(cloud_bucket_name)
	bucket.storage_class = "COLDLINE"
	new_bucket = storage_client.create_bucket(bucket, location="us")
	print(
		"\nCreated bucket {} in {} with storage class {}\n".format(
			new_bucket.name, new_bucket.location, new_bucket.storage_class
		)
	)


def upload_to_cloud(bucket_name, local_file_path, path_to_destination_blob):
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(path_to_destination_blob)
	blob.upload_from_filename(local_file_path)
	print(
		f"File {local_file_path} uploaded to {path_to_destination_blob}\n"
	)


def transcript_audio(bucket_name, path_to_file):
	file_uri = "gs://" + bucket_name + '/' + path_to_file
	audio_file = speech.RecognitionAudio(uri=file_uri)
	config = speech.RecognitionConfig(
		sample_rate_hertz = 48000,
		enable_automatic_punctuation = True,
		language_code = 'en-US',
		use_enhanced = True
	)
	response = speech_client.recognize(config=config, audio=audio_file)

	for result in response.results:
		print("Transcript: {}".format(result.alternatives[0].transcript))
		"""
		with open('transcript.txt', 'w') as f:
			f.write(result.alternatives[0].transcript)
		return (result.alternatives[0].transcript)
		"""
		res = result.alternatives[0].transcript
		if res[-1] != '.':
			res += '.'
		return res


def transcript_pdf(file_path):
	pdfFileObj = open(file_path, 'rb')
	pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

	pageObj = pdfReader.getPage(0)
	res = pageObj.extractText()
	pdfFileObj.close()
	return res

def youtube_to_mp3(link: str, filename):
	yt = YouTube(link)
	video = yt.streams.filter(only_audio=True).first()
	outfile = video.download(output_path='./inputs/')
	os.rename(outfile, filename)


main()