from elevenlabs import set_api_key,generate,save
import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import *
import glob
import cv2
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import moviepy.editor as mp #for mov to mp3 conversion
from deepface import DeepFace
import torch
import json
from pydub import AudioSegment
from random import randint
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
from moviepy.audio.fx.all import volumex
import moviepy.editor as mp

with open("../eleven.pass", 'r') as file:
    set_api_key(file.read())

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_FILENAME = "tmp.jpg"

old_prompt = """
 have a 10 second video clip which has been proccessed by CV and ML stuff into this. 
 there is a transcript of what was said, and descriptions of what is detected by the yolo model and image captioner ever 2 seconds. 
 generate a cohesive funny story voiceover of what is happening to play over the 10s clip. it needs to be a very brief voiceover up to only 34 words
"""

#  have a series of 10 second video clips which have been proccessed by CV and ML stuff into list of data for each clip. 
#  for each clip, there is a transcript of what was said, and descriptions of what is detected by the yolo model and image captioner ever 2 seconds. 
#  your task is to decide whether to 3rd person voiceover each clip, in which case you generate a transcript, or whether to leave the existing voice in place, in which case you put NOTHING.
#  you will generate a cohesive funny storyline of what is happening to decide this.

#  you will output a list separated by n ewlines and NOTHING else. each element of the list corresponds to the voiceover for the clip, to play over each 10s clip, and if you do not wish to voiceover the clip, the element should be NOTHING.
#  each voiceover can be up to 50 words to explain, be descriptive. make up feelings, use metaphors, use your imagination!!
#  here is an example output for an example with 6 video clips where 3 were chosen for voiceover, and 3 are left with original audio:
# =========================

# P
prompt = """
you are voiceovering various segments of a video to make a story. the prompt has the number of lines in the list and a list of dictionaries which contain metadata for sequential 10 second video clips. the numbers of lines of your output must match the length of the prompt list (each video gets one line of voiceover / NOTHING if you choose not to voiceover that video)

PROMPT:

6
[{'transcript': 'BUZZING', 'scene_descriptions': [{'items': ['person', 'person', 'person', 'person', 'person', 'backpack', 'mouse'], 'description': 'a group of people standing around a room', 'face': None}, {'items': ['person', 'person'], 'description': 'a woman walking down a hallway in a hospital', 'face': None}]}, {'transcript': "What's your favorite kind of pasta, Nathan? I like tortellinis. Tortellinis? Not really pasta, though. ", 'scene_descriptions': [{'items': ['person', 'person'], 'description': 'a man standing in a room with a cell phone', 'face': None}, {'items': ['person', 'person'], 'description': 'a man in a gray shirt', 'face': None}]}, {'transcript': "You have to go around this way, the line's too big. Woah. Okay, that's a wrap.", 'scene_descriptions': [{'items': ['cat'], 'description': 'a woman is standing in a room with a mirror', 'face': None}, {'items': ['tv', 'person'], 'description': 'a group of people sitting around a table', 'face': None}]}, {'transcript': "I think it's going to be a little higher, like an eye level. Eye level? Isn't this a", 'scene_descriptions': [{'items': ['person', 'person', 'refrigerator'], 'description': 'a white wall with drawings on it', 'face': None}, {'items': ['person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'dining table'], 'description': 'a group of people standing in a room', 'face': None}]}, {'transcript': "It's supposed to be like here. But it's going to be like, ideally, producing like, forward, perfectly forward all the time. Perfectly forward? It would be less ground, minimizing the amount of ground.", 'scene_descriptions': [{'items': ['person', 'dog'], 'description': 'a man is playing with a video game', 'face': None}, {'items': ['person', 'person', 'person'], 'description': 'a man in a blue shirt', 'face': None}]}, {'transcript': "Oh, okay, okay. I'll just lean it up a bit. There's the TV.", 'scene_descriptions': [{'items': ['person', 'person', 'person'], 'description': 'a group of people are sitting at tables', 'face': None}, {'items': ['tv', 'person', 'chair'], 'description': 'a man is standing in a room with a computer', 'face': None}]}]

 ================

 OUTPUT:

 A group gathers around tables, conversation buzzing. In a hospital, a woman strolls down a hallway.
"What's your favorite pasta, Nathan?" A man stands with a cell phone. Another in a gray shirt listens.
NOTHING
In a room, a woman gazes at a mirror, a group discusses around a table.
NOTHING
Playing a video game, a man interacts with a dog. In a blue shirt, another contemplates.

==================

PROMPT:
"""

prompt = """

we are writing a 6 part screenplay loosely based on the above transcript of a real audio transcript. The screenplay will have 6 distinct scenes, each 10 seconds long. For 3 of these scenes, you will write a 3rd person limited narration. For the remaining 3, the original real audio will be kept. Select which 3 are narrated over, and which 3 are kept, to your liking. Ensure that the entire 6 part scene has a coherent and interesting storyline, full of exciting twists and turns that will entertain the audience! Consider drama such as a divorce, a murder, a spy noir film, an affair, or a robbery, easter egg hunt, coding homework, swimming, hackathon, eating your mom, dying, living, exploding, barbenheimer, petting cats, smelling flowers, wearing hard wear, baking bread, yeast, construction working, vermeer, skydiving, walking on grass, touching grass, doing your mom, happening. However, don't choose more than one and overcomplicate the story.
Also, you DO NOT always have to choose one!! Make up your own creative scenario.
However, note that the storyline MUST ALWAYS be coherent. The storyline should be very easy to follow.
Include transition words when possible. Each scenario or scene has to transition to the next scene in a way that makes sense.
The story must have CONTINUITY so remember not to include too many scenarios at once.
Here are more concrete limitations:
For the 3 scenes that have the 3rd person narration, they should only be 1-2 sentences long.
For all 6 scenes, format your output for each scene as an element, inside of a Python list of 6 elements. Each element will have 2 elements nested inside, which will be NARRATION or TRANSCRIPT and then the actual text. For example, I may have element 1 be ["narration", "A bustling hospital hallway. A man named Nathan faces a daunting line."] 

Format output as follows, WITH CORRECT PYTHON SYNTAX FOR A NESTED LIST:
[["xxx", "xxx"],
    ["xxx", "xxx"],
    ["xxx", "xxx"],
    ["xxx", xxx"],
    ["xxx", "xxx"],
    ["xxx", "xxx"]]

DO NOT output any additional code, comments, words, symbols, or anything at all. Keep the output very strict to the provided format above. ONLY RESPOND WITH THE OUTPUT.
output:


"""


with open("../openai.pass", 'r') as file:
    openai.api_key = file.read()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def save_pil(image, filename = DEFAULT_FILENAME):
    image.save(filename)

'''
Takes in a PIL Image and path to database of named reference face images. 
Image names in the database should match names of the people.

Returns name of face if it finds one. 
'''
def id_face(pil_img, db_path):
    save_pil(pil_img)
    dfs = DeepFace.find(img_path = DEFAULT_FILENAME, db_path = db_path, enforce_detection=False)[0]
    if not dfs.shape[0]:
        return None
    return [dfs.sort_values("VGG-Face_cosine")["identity"][i].split("/")[-1][:-4] for i in range(dfs.shape[0])]


'''
Takes in a PIL Image.
Returns top emotion of face if there is a face. If there is no face, returns None.
'''
def id_emotion(pil_img):
    save_pil(pil_img)
    try:
        emotion = DeepFace.analyze(img_path = DEFAULT_FILENAME, actions = ['emotion'])
        return max(emotion[0]['emotion'].keys(), key = lambda k: emotion[0]['emotion'][k])
    except:
        return None # No face

def find_newest_file(folder_path):
    list_of_files = glob.glob(os.path.join(folder_path, '*.MOV'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_top_objects(frame_results, top_n=3, confidence_threshold=0.8):
    filtered_results = [obj for obj in frame_results if obj['confidence'] > confidence_threshold]
    sorted_results = sorted(filtered_results, key=lambda x: x['confidence'], reverse=True)
    return sorted_results[:top_n]



def extract_frames(video_path, output_folder, frame_interval=120):
    # speech to text

    input_mov_file = video_path
    output_mp3_file = "raw_mom.mp3"
    video_clip = mp.VideoFileClip(input_mov_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_mp3_file, codec="mp3")
    video_clip.close()
    audio_clip.close()

    print(f"Conversion to MP3 complete. MP3 file saved as '{output_mp3_file}'.")


    #using whisper
    audio_file= open(output_mp3_file, "rb")
    transcript = (openai.Audio.transcribe("whisper-1", audio_file)).text
    list_of_lists_of_things = []

    print(transcript)

    if True or len(transcript) < 60:
        cap = cv2.VideoCapture(video_path)

        frame_count = 0 
        images = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_interval == 0:
                images.append(frame)

        cap.release()
        cv2.destroyAllWindows()

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        list_of_lists_of_things = []

        for i, image in enumerate(images):
            model = YOLO("yolov8n.pt")
            results = model.predict(source=image, save_conf=True)
            confs = results[0].boxes.conf
            things = [model.names[int(c)] for i,c in enumerate(results[0].boxes.cls) if float(confs[i])>0.5]

            # make raw_image the object u want
            raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_id=None
            # face_id = id_face(Image.fromarray(image), "shrockers/")

            # unconditional image captioning
            inputs = processor(raw_image, return_tensors="pt")


            out = blip_model.generate(**inputs)
            desc = processor.decode(out[0], skip_special_tokens=True)


            list_of_lists_of_things.append({"items":things, "description":desc})
        
    final_data  = {"transcript" : transcript, "scene_descriptions":list_of_lists_of_things}
    print(final_data)
    return final_data

def split_into_10s(input_folder, output_folder):
    
    # clear output folder
    import os, shutil
    folder = output_folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    # parse in
    newest_file = find_newest_file(input_folder)

    clip = VideoFileClip(newest_file)

    # Get the duration of the video in seconds
    duration = clip.duration

    # Calculate the number of 10-second clips
    num_clips = int(duration // 10)

    # Create MOM10s folder if it doesn't exist
    output_folder_path = os.path.join(os.getcwd(), output_folder)
    os.makedirs(output_folder_path, exist_ok=True)

    # Split the video into 10-second clips
    for i in range(num_clips):
        start_time = i * 10
        end_time = (i + 1) * 10
        subclip = clip.subclip(start_time, end_time)
        subclip.write_videofile(os.path.join(output_folder_path, f"clip_{i + 1}.mp4"), codec="libx264", threads=4)

    clip.reader.close()
    clip.audio.reader.close_proc() 

def get_frames(folder):
    """
    Get a list of OpenCV images representing the first frame of each mp4 file in the specified folder.

    Parameters:
    - folder (str): The path to the folder.

    Returns:
    - list: A list of OpenCV images.
    """
    image_list = []

    try:
        # Get the list of files in the folder
        files = [f for f in os.listdir(folder) if f.endswith('.mp4')]

        # Iterate through each mp4 file
        for file in files:
            file_path = os.path.join(folder, file)

            # Open the video file
            cap = cv2.VideoCapture(file_path)

            # Read the first frame
            ret, frame = cap.read()

            # Append the first frame to the list
            if ret:
                image_list.append(frame)

            # Release the video capture object
            cap.release()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return image_list

def make_background_audio(voiceover):
    voiceover = ' '.join(voiceover)

    prompt = voiceover + """\nGiven the above text descriptions, analyze the situation and tone to find the most appropriate adjective.
        Available adjectives are [1]'action', [2]'reflective and happy', [3]'creepy or mischevious', [4]'energetic!',
        [5]'fun or jazzy', [6]'nostalgic', [7]'sad and rainy', [8]'gentle and uplifting', [9]'technology or mysterious and intriguing',
        [10]'upbeat and funky', [11]'upbeat and lighthearted and happy', [12]'extremely mysterious'. Give the answer as a single digit
        based on the given song indexes. Do not give any answer other than a single digit without brackets.
        For example, your output can be 3. output:"""


    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=2000
    )

    message = response['choices'][0]['text']
    
    genre_key = int(''.join([e for e in message if e in '1234567890']))
    genre_key = genre_key if genre_key < 13 and genre_key > 0 else 6

    song_dict = {1: 'action.mp3',
                2: 'happy_reflective.mp3',
                3: 'creepy_mischeivious.mp3',
                4: 'energetic.mp3',
                5: 'fun_jazz.mp3',
                6: 'nostalgic.mp3',
                7: 'sad_rainy.mp3',
                8: 'gentle_uplifting.mp3',
                9: 'technology_mystery_intrigue_upbeat.mp3',
                10: 'upbeat_funky.mp3',
                11: 'upbeat_lighthearted_happy.mp3',
                12: 'very_mysterious.mp3'}

    #make a copy of the song you want and name it background_music.mp3
    print("../audio_assets/"+song_dict.get(genre_key))
    print(os.path.exists("../audio_assets/"+song_dict.get(genre_key)))
    background_audio = AudioFileClip(f"../audio_assets/"+song_dict.get(genre_key))
    # background_audio.export("background_music.mp3", format="mp3")
    return background_audio


def apply_frame_filter(frame, color, original_weight):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Apply color grading to the grayscale frame
    colored_frame = cv2.applyColorMap(gray_frame, color)
    # Combine the colored frame with the original frame
    result_frame = cv2.addWeighted(frame, original_weight, colored_frame, 1 - original_weight, 0)
    return result_frame


def apply_general_filter(video_clip, color, original_weight, lum, blackwhite): #lower lum num = darker
    #step1: tint
    if (not blackwhite):
        filtered_frames = [apply_frame_filter(frame, color, original_weight) for frame in video_clip.iter_frames(fps=video_clip.fps)]
        filtered_clip = mp.ImageSequenceClip(filtered_frames, fps=video_clip.fps) #add clips together to make vid
        filtered_clip = filtered_clip.set_audio(video_clip.audio) #fix audio
    
    else:
         filtered_clip = filtered_clip.fx(vfx.blackwhite)
    
    #step 2: luminosity
    filtered_clip = filtered_clip.fx(vfx.colorx, lum)

    return filtered_clip


def choose_filter(voiceover):
    voiceover = ' '.join(voiceover)
    

    prompt = voiceover + """\nGiven the above text transcripts, analyze the situation and tone to find the most appropriate filter for the video.
        Available adjectives are [1]'spooky mystery y2k', [2]'royal historical ball', [3]'dark night', [4]'faded memories',
        [5]'dirty yellow' [6]'aesthetic film' [7]'spy movie noir'. Give the answer as a single digit
        based on the given song indexes. Do not give any answer other than a single digit without brackets.
        For example, your output can be 3. output:"""
    
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=200
    )

    message = response['choices'][0]['text']
    
    genre_key = int(''.join([e for e in message if e in '1234567890']))
    genre_key = genre_key if genre_key < 7 and genre_key > 0 else 6



    filter_dict = {1: [cv2.COLORMAP_DEEPGREEN,0.6,0.6,False],
                2: [cv2.COLORMAP_OCEAN,0.6,0.3,False],
                3: [cv2.COLORMAP_BONE, .9,.3,False],
                4: [cv2.COLORMAP_SUMMER,.6,1,False],
                5: [cv2.COLORMAP_DEEPGREEN, .9,1,False],
                6: [cv2.COLORMAP_VIRIDIS, .9,.6,False],
                7: [None,None, .8,True]
                }
    return filter_dict[genre_key]
    







def get_files(output_folder):
    gpt_query = ""
    for i,image in enumerate(get_frames(output_folder)):

        raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_id=None
        # face_id = id_face(Image.fromarray(image), "shrockers/")

        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")

        # out = blip_model.generate(**inputs)
        # desc = processor.decode(out[0], skip_special_tokens=True)
        # gpt_query += f"{i}: {desc}\n"

        out = blip_model.generate(**inputs)
        desc = processor.decode(out[0], skip_special_tokens=True)
        gpt_query += f"{i+1}: {desc}\n"

    gpt_query += "\n\n give me a comma separated list of numbers of the top 6 most interesting description numbers from the above list. DO NOT SAY ANYTHING ELSE EXCEPT THE ANSWER. YOUR ANSWER MUST BE FORMATTED AS A COMMA SEPARATED LIST OF 6 NUMBERS"

#     gpt_query = """0: a man in a blue shirt
# 1: a man in a white shirt
# 2: a man in a blue shirt is standing in a room
# 3: a man is walking through a hallway in a building
# 4: a man in a blue shirt and white pants
# 5: a man in a gray shirt
# 6: a man with curly hair
# 7: a man in a gray shirt and black pants
# 8: a blur of people walking through a building
# 9: a white desk with a monitor on it
# 10: a man in a room with a guitar
# 11: a man in a white shirt is walking up a stair
# 12: a blur of people walking down a hospital hallway
# 13: a group of people are walking through a building
# 14: a group of people are walking around a room
# 15: a man in a gray sweater
# 16: a blur of a person walking down a hospital hallway
# 17: a young boy is walking down a hallway


# #  give me a comma separated list of numbers of the top 6 most interesting descriptions that make a good storyline together. DO NOT SAY ANYTHING ELSE EXCEPT THE ANSWER. YOUR ANSWER MUST BE FORMATTED LIKE THIS WHERE <number> is a number like 1 or 2 or 3 etc:  <number>,<number>,<number>,<number>,<number>,<number>
# # """
    print(gpt_query)

    # Make a call to the OpenAI API
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=gpt_query,
    max_tokens=2000
    )

    # Extract the generated message
    message = response['choices'][0]['text']
    print(message)

    numbers = [''.join([r for r in e if r in '1234567890']) for e in message.split(',')]
    print(numbers)
    files = [f"clip_{i}.mp4" for i in numbers if i]
    return files


def add_sound_effects(video_clip, final_data):
    # video_clip = VideoFileClip("../MOM10s/clip_1.mp4")
    gpt_query = """
    have a 10 second video clip which has been proccessed by CV and ML stuff into this. 
    there is a transcript of what was said, and descriptions of what is detected by the yolo model and image captioner ever 2 seconds. 
    """
    gpt_query += "\n" + str(final_data) + "\n"
    gpt_query += """
    Now here is a list of some sound effects to choose from:
    """
    
    gpt_query += "\n".join(os.listdir("../audio_assets/sfx"))

    gpt_query += "Based on the given information, what sound effect should I use to complement the video? Respond with the filename ONLY, and nothing else."

    print(gpt_query)

    # Make a call to the OpenAI API
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=gpt_query,
    max_tokens=3000
    )

    # Extract the generated message
    message = response['choices'][0]['text']
    message = message.strip()
    print(f"../audio_assets/sfx/{message}")

    try:
        sound_effect = AudioFileClip(f'../audio_assets/sfx/{message}')
    except:
        sound_effect = AudioFileClip(f'../audio_assets/sfx/jazz.mp3')
        
    sound_effect = volumex(sound_effect, 0.25)
    final_audio = CompositeAudioClip([video_clip.audio, sound_effect.set_start(randint(0,5))]) #replac
    video_clip = video_clip.set_audio(final_audio)
        

    # video_clip.write_videofile("static/test.mp4")
    return video_clip    

def generate_video():
    
    print("here")
    
    input_folder = "../MOMents"
    output_folder = "../MOM10s"

    #split_into_10s(input_folder, output_folder) #TODO uncomment if using new video


    # curr_clip = "MOM10s/clip_6.mp4"

    interesting_files = get_files(output_folder)
    final_data_list = []

    clips = []
    for curr_clip in interesting_files:
        curr_clip = output_folder + "/" +curr_clip
        if os.path.exists(curr_clip):
            final_data = extract_frames(curr_clip, output_folder)
            final_data_list.append(final_data)
    
    print(final_data_list)

    # final_data_list = [{'transcript': 'What was that?', 'scene_descriptions': [{'items': ['person'], 'description': 'a man is sitting down on a laptop'}, {'items': ['person'], 'description': 'a man with brown hair'}, {'items': [], 'description': 'a blur of a person in a room'}, {'items': ['person'], 'description': 'a man with a blue shirt'}, {'items': ['person', 'person'], 'description': 'a man sitting on a chair'}]}, {'transcript': 'Okay, guys, we are at 30 seconds of video. And wait, how long is the movie going to be?', 'scene_descriptions': [{'items': [], 'description': 'a large machine is moving through a factory'}, {'items': [], 'description': 'a blur of a person walking down a hallway'}, {'items': ['person', 'person'], 'description': 'two people sitting in chairs in a room'}, {'items': ['person', 'person', 'laptop'], 'description': 'a group of people sitting around a table with laptops'}, {'items': [], 'description': 'a blur of people on a train'}]}, {'transcript': "Oh, that's pretty delicious. Bubble, what do you have to say?", 'scene_descriptions': [{'items': [], 'description': "a close up of a person's face with a yellow and black hair"}, {'items': ['person'], 'description': 'a person is cutting a cake on a table'}, {'items': ['teddy bear'], 'description': 'a person is holding a stuffed animal'}, {'items': ['teddy bear'], 'description': 'a stuffed animal is being fed by a person'}, {'items': ['person'], 'description': 'a man is seen in the middle of a restaurant'}]}, {'transcript': 'Thank you for watching the video.', 'scene_descriptions': [{'items': [], 'description': 'a long hallway with a blue and yellow line on the floor'}, {'items': [], 'description': 'a blur of a door in a hallway'}, {'items': [], 'description': 'a person is holding a knife in their hand'}, {'items': [], 'description': 'a bed with a yellow pillow and a white wall'}, {'items': [], 'description': 'a door is open in an office'}]}, {'transcript': "Thank you for watching and don't forget to subscribe!", 'scene_descriptions': [{'items': ['refrigerator'], 'description': 'a white wall with a sign on it'}, {'items': [], 'description': 'a man is walking down a hallway in an office'}, {'items': [], 'description': 'a long hallway with a door and a person walking down the hallway'}, {'items': ['refrigerator', 'refrigerator'], 'description': 'a long hallway with a black floor and white walls'}, {'items': [], 'description': 'a hallway with a wooden floor and a yellow door'}]}, {'transcript': "That's like the one place you shouldn't be. There was no one in there. Bro, that's like actually the worst.", 'scene_descriptions': [{'items': ['person', 'person', 'person'], 'description': 'a group of people sitting in a room'}, {'items': ['laptop', 'person'], 'description': 'a group of people sitting around a laptop'}, {'items': ['person', 'person', 'laptop'], 'description': 'a man sitting on a chair with a laptop'}, {'items': ['cell phone'], 'description': 'a blur of people on an airplane'}, {'items': ['person'], 'description': 'a man is seen in the middle of a room with a woman in the middle'}]}]

    response = openai.Completion.create(engine="text-davinci-003",prompt=  str(json.dumps(final_data_list)) + "\n" + prompt,max_tokens=2000,presence_penalty=1)
    #prompt + "\n\n" + str(len(final_data_list)) + "\n" + str(json.dumps(final_data_list))
    message = response['choices'][0]['text'].strip()

    if message[-5:].count(']') != 2:
        message+=']'
    print(message)

    voiceovers = [e[1] if e[0]=="narration" else "NOTHING" for e in eval(message)]
    

    # voiceovers = [e for e in message.split("\n") if len(e)>2 and "output:" not in e.lower()]
    if len(voiceovers) > len(interesting_files):
        voiceovers = voiceovers[0:len(interesting_files)]

    lengths = [(c,len(e["transcript"])) for c,e in enumerate(final_data_list)]
    lengths.sort(key = lambda x: x[1])
    no_voiceover = [lengths[0][0], lengths[1][0]]

    for index,voiceover in enumerate(voiceovers):
        audio_clip = 0
        curr_clip = output_folder + "/" + interesting_files[index]

        if "NOTHING" not in voiceover: #and index not in no_voiceover:
            audio = generate(
            text=voiceover,
            voice="Harry",
            model="eleven_multilingual_v2"
            )

            print(voiceover)

            save(
                audio,  "output_eleven.wav"
            )

            audio_clip = AudioFileClip("output_eleven.wav")

            audio_clip = audio_clip.subclip(0, min([11,audio_clip.duration]))

        video_clip = VideoFileClip(curr_clip)
        if audio_clip:
            video_clip = video_clip.set_audio(audio_clip)
        
        video_clip = add_sound_effects(video_clip, " ".join(voiceover))

        clips.append(video_clip)

    final_clip = concatenate_videoclips(clips)
    
    #adding background audio
    audio_background = make_background_audio(message) #background audio
    audio_background = audio_background.subclip(0, min([60,audio_background.duration]))
    audio_background = volumex(audio_background, 0.2)
    final_audio = CompositeAudioClip([final_clip.audio, audio_background]) #add in background audio to create final audio
    final_clip = final_clip.set_audio(final_audio) #set audio to final audio

    params = choose_filter(voiceovers)
    final_clip = apply_general_filter(final_clip, params[0], params[1], params[2], params[3])



    final_clip.write_videofile("static/running5.mp4")


generate_video()