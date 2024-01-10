import numpy as np
from moviepy import *
from moviepy.editor import *
from moviepy.video.tools.segmenting import findObjects
from PIL import Image
import os
import openai

read_config = open('settings.config','r').read()
the_openai_api = eval(read_config)['openai_api_key']

#set the api for openai

openai.api_key = the_openai_api




#information gathering function for selected video
def video_info(path):
	sel_video = VideoFileClip(path)
	duration = sel_video.duration #duration in seconds
	total_frames = sel_video.reader.nframes  #total frames in the whole video
	fps = sel_video.reader.fps
	size = sel_video.size

	return {"main_video":sel_video,"duration":duration,"total_frames":total_frames,"fps":fps,"size":size}




#cliparray to final video renderer function
def clip_add_and_save(clip_array,fps,duration,video_size,filename):

	overlay = CompositeVideoClip(clip_array,size=video_size)
	overlay=overlay.set_duration(duration)
	overlay=overlay.set_fps(fps)
	overlay.write_videofile(f"output/{filename}.mp4")

	return True


#read the CTA or promo text from 'promotional_text.txt' file to include it at the end of the video
def read_promotional_text():
	fopen = open('promotional_text.txt','r')

	all_lines = fopen.read().splitlines()

	fopen.close()

	return all_lines



#ai function that receives a promt and give output according to that
def read_ai_question_and_answer(promt):

	response = openai.Completion.create(
	  model="text-davinci-002",
	  prompt=promt,
	  temperature=0.7,
	  max_tokens=35,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0
	)

	ai_response = response['choices'][0]['text'].strip()

	
	return str(ai_response)



#this function makes a textClip object based on provided informations
def textclip_maker(text,width,fps):


    textclip = TextClip(text,
                        fontsize=59,color="white",bg_color="white",align="center",stroke_color="black",
                        stroke_width=4, kerning=5,font="Lane",size=(width,0),method='caption'
                        )
    #textclip=textclip.set_duration(duration)


    textclip=textclip.set_fps(fps)

    textclip=textclip.set_position(("center",250))

    return textclip



def make_video(lines_array,words_for_second,width,fps):
    all_textClips = []
    total_words = 0
    for line in lines_array:
        total_words+=len(line.split(' '))
        
    total_video_duration = float(total_words/words_for_second)
        
    start_timer = 0
    
    for line in lines_array:
        line_duration = float(len(line.split(' '))/words_for_second)
        all_words = line.split(' ')
        width = width
        createTextClip = textclip_maker(line,width,fps)
        if line in ["details in bio" , "Check details in bio" , "Details In Bio" , "Details in my Bio" , "Details in bio" , "Click the link in bio" , "link in bio" , "check bio" , "Check bio to know more"]:
        	createTextClip = createTextClip.set_duration(3)
        else:
        	createTextClip = createTextClip.set_duration(line_duration)

        createTextClip = createTextClip.set_start(start_timer)
        createTextClip = createTextClip.set_position(('center',290))
        
        all_textClips.append(createTextClip)
        print('start timer is ',start_timer)
        
        start_timer+=line_duration
        print('second start timer - ',start_timer)

    for x in all_textClips:
    	print('duration is ', x.duration)

    
    return all_textClips









def main_func():
	promo_text = read_promotional_text()
	get_main_video = input("Main video file path ex:(input/video.mp4) :  ")

	is_ai_mode = input("Is on AI mode ? yes/no : ")



	vid = video_info(get_main_video)
	vid_inst = vid['main_video']
	vid_fps = vid['fps']


	if is_ai_mode == True or is_ai_mode.lower() == 'yes' :
		
		op = open('questions.txt','r').read().splitlines()

		#prompt = input("give AI a promt example-(write a question about something) : ")
		#video_no = input("how many video you need ? :  ")
		
		counter = 0
		for i in op:

			ai_list = []
			file_name = i.replace(' ','_')
			file_name = file_name.replace('?','-')
			
			ai_res = read_ai_question_and_answer(i+' must give the anwer in less than 30 words and no ackward sentence ending')
			ai_list.append(i)
			ai_list.append(ai_res.strip())

			promo_list = read_promotional_text()

			for promo_line in promo_list:
				ai_list.append(promo_line)

			print(ai_list)

			theTextClips = make_video(ai_list,2.7,800,vid_fps)
			comp_text = CompositeVideoClip(theTextClips,size=(1080,1920))
			
			comp_vid = CompositeVideoClip([vid_inst,comp_text],size=(1080,1920))
			comp_vid = comp_vid.set_duration(comp_text.duration)
			comp_vid.write_videofile(f"output/{file_name}.mp4")
			print(f"Video produced {i} successfully ")

			counter+=1




	else:
		

		theTextClips = make_video(promo_text,2,800,vid_fps)

		comp_text = CompositeVideoClip(theTextClips,size=(1080,1920))

		

		comp_vid = CompositeVideoClip([vid_inst,comp_text],size=(1080,1920))
		comp_vid = comp_vid.set_duration(comp_text.duration)
		

		comp_vid.write_videofile("output/manual_vid.mp4")

		print("Video produced")



main_func()