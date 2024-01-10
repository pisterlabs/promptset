import openai
from pytube import YouTube
import argparse
import os
import whisper
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--link", help="youtube video link here", default=None, type=str, required=False)
parser.add_argument("--video_file", help="local video path here", default=None, type=str, required=False)
parser.add_argument("--audio_file", help="local audio path here", default=None, type=str, required=False)
parser.add_argument("--srt_file", help="srt file input path here", default=None, type=str, required=False)  # New argument
parser.add_argument("--download", help="download path", default='./downloads', type=str, required=False)
parser.add_argument("--output_dir", help="translate result path", default='./results', type=str, required=False)
parser.add_argument("--video_name", help="video name, if use video link as input, the name will auto-filled by youtube video name", default='placeholder', type=str, required=False)
parser.add_argument("--model_name", help="model name only support text-davinci-003 and gpt-3.5-turbo", type=str, required=False, default="gpt-3.5-turbo")
parser.add_argument("-only_srt", help="set script output to only .srt file", action='store_true')
parser.add_argument("-v", help="auto encode script with video", action='store_true')
args = parser.parse_args()



# input should be either video file or youtube video link.
if args.link is None and args.video_file is None and args.srt_file is None and args.audio_file is None:
    print("need video source or srt file")
    exit()

# set up
openai.api_key = os.getenv("OPENAI_API_KEY")
DOWNLOAD_PATH = args.download
if not os.path.exists(DOWNLOAD_PATH):
    os.mkdir(DOWNLOAD_PATH)
    os.mkdir(f'{DOWNLOAD_PATH}/audio')
    os.mkdir(f'{DOWNLOAD_PATH}/video')

RESULT_PATH = args.output_dir
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

VIDEO_NAME = args.video_name
model_name = args.model_name

# get source audio
if args.link is not None and args.video_file is None and args.audio_file is None:
    # Download audio from YouTube
    video_link = args.link
    video = None
    audio = None
    try:
        yt = YouTube(video_link)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if video:
            video.download(f'{DOWNLOAD_PATH}/video')
            print('Video download completed!')
        else:
            print("Error: Video stream not found")
        audio = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        if audio:
            audio.download(f'{DOWNLOAD_PATH}/audio')
            print('Audio download completed!')
        else:
            print("Error: Audio stream not found")
    except Exception as e:
        print("Connection Error")
        print(e) 
        exit()
    
    video_path = f'{DOWNLOAD_PATH}/video/{video.default_filename}'
    audio_path = '{}/audio/{}'.format(DOWNLOAD_PATH, audio.default_filename)
    audio_file = open(audio_path, "rb")
    if VIDEO_NAME == 'placeholder':
        VIDEO_NAME = audio.default_filename.split('.')[0]

elif args.video_file is not None:
    # Read from local
    video_path = args.video_file
    if args.audio_file is not None:
        audio_file= open(args.audio_file, "rb")
        audio_path = args.audio_file
    else:
        os.system(f'ffmpeg -i {args.video_file} -f mp3 -ab 192000 -vn {DOWNLOAD_PATH}/audio/{VIDEO_NAME}.mp3')
        audio_file= open(f'{DOWNLOAD_PATH}/audio/{VIDEO_NAME}.mp3', "rb")
        audio_path = f'{DOWNLOAD_PATH}/audio/{VIDEO_NAME}.mp3'

if args.audio_file is not None:
    audio_file= open(args.audio_file, "rb")
    audio_path = args.audio_file

if not os.path.exists(f'{RESULT_PATH}/{VIDEO_NAME}'):
    os.mkdir(f'{RESULT_PATH}/{VIDEO_NAME}')

# Instead of using the script_en variable directly, we'll use script_input
srt_file_en = args.srt_file 
if srt_file_en is not None: 
    with open(srt_file_en, 'r', encoding='utf-8') as f:
        script_input = f.read()
else:
    # using whisper to perform speech-to-text and save it in <video name>_en.txt under RESULT PATH.
    srt_file_en = "{}/{}/{}_en.srt".format(RESULT_PATH, VIDEO_NAME, VIDEO_NAME)
    if not os.path.exists(srt_file_en):
        # use OpenAI API for transcribe
        # transcript = openai.Audio.transcribe("whisper-1", audio_file) 

        # use local whisper model 
        model = whisper.load_model("base") # using base model in local machine (may use large model on our server)
        transcript = model.transcribe(audio_path)

        #Write SRT file
        from whisper.utils import WriteSRT

        with open(srt_file_en, 'w', encoding="utf-8") as srt:
            writer = WriteSRT(RESULT_PATH)
            writer.write_result(transcript, srt)

    # split the video script(open ai prompt limit: about 5000)
    with open(srt_file_en, 'r', encoding='utf-8') as f:
        script_en = f.read()
        script_input = script_en

if not args.only_srt:
    from srt2ass import srt2ass
    assSub_en = srt2ass(srt_file_en, "starPigeon", "No", "Modest")
    print('ASS subtitle saved as: ' + assSub_en)

# force translate the starcraft2 term into chinese according to the dict
# TODO: shortcut translation i.e. VA, ob
# TODO: variety of translation
from csv import reader
import re

# read dict
with open("finetune_data/dict.csv",'r', encoding='utf-8') as f:
  csv_reader = reader(f)
  term_dict = {rows[0]:rows[1] for rows in csv_reader}

def clean_timestamp(lines):
  new_lines = []
  strinfo = re.compile('[0-9]+\n.{25},[0-9]{3}')   
  new_lines = strinfo.sub('_-_', lines)
  print(new_lines)
  return new_lines


ready_lines = re.sub('\n', '\n ', script_input)
ready_words = ready_lines.split(" ")
i = 0
while i < len(ready_words):
  word = ready_words[i]
  if word[-2:] == ".\n" :
    if word[:-2].lower() in term_dict :
      new_word = word.replace(word[:-2], term_dict.get(word[:-2].lower())) + ' '
      ready_words[i] = new_word
    else :
      word += ' '
      ready_words[i] = word
  elif word.lower() in term_dict :
      new_word = word.replace(word,term_dict.get(word.lower())) + ' '
      ready_words[i] = new_word
  else :
    word += " "
    ready_words[i]= word
  i += 1

script_input_withForceTerm = re.sub('\n ', '\n', "".join(ready_words))


# Split the video script by sentences and create chunks within the token limit
n_threshold = 1000  # Token limit for the GPT-3 model
script_split = script_input_withForceTerm.split('\n')

script_arr = []
script = ""
for sentence in script_split:
    if len(script) + len(sentence) + 1 <= n_threshold:
        script += sentence + '\n'
    else:
        script_arr.append(script.strip())
        script = sentence + '\n'
if script.strip():
    script_arr.append(script.strip())

# Translate and save
for s in tqdm(script_arr):
    # using chatgpt model
    if model_name == "gpt-3.5-turbo":
        # print(s + "\n")
        response = openai.ChatCompletion.create(
            model=model_name,
            messages = [
                {"role": "system", "content": "You are a helpful assistant that translates English to Chinese and have decent background in starcraft2."},
                {"role": "system", "content": "Your translation has to keep the orginal format and be as accurate as possible."},
                {"role": "system", "content": "You are not allowed to add any comments or notes."},
                {"role": "system", "content": '''All your translations have to be in the format below:
                
                <id number int>
                <Start Time Stamp> --> <End Time Stamp>
                <Translated Text>
                
                '''},
                {"role": "user", "content": 'Translate the following English text to Chinese: "{}"'.format(s)}
            ],
            temperature=0.15
        )
        with open(f"{RESULT_PATH}/{VIDEO_NAME}/{VIDEO_NAME}_zh.srt", 'a+') as f:
            f.write(response['choices'][0]['message']['content'].strip())
            f.write("\n")

    if model_name == "text-davinci-003":
        prompt = f"Please help me translate this into Chinese:\n\n{s}\n\n"
        # print(prompt)
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        with open(f"{RESULT_PATH}/{VIDEO_NAME}/{VIDEO_NAME}_zh.srt", 'a+') as f:
            f.write(response['choices'][0]['text'].strip())
            f.write("\n")

if not args.only_srt:
    assSub_zh = srt2ass(f"{RESULT_PATH}/{VIDEO_NAME}/{VIDEO_NAME}_zh.srt", "starPigeon", "No", "Modest")
    print('ASS subtitle saved as: ' + assSub_zh)

if args.v:
    if args.only_srt:
        os.system(f'ffmpeg -i {video_path} -vf "subtitles={RESULT_PATH}/{VIDEO_NAME}/{VIDEO_NAME}_zh.srt" {RESULT_PATH}/{VIDEO_NAME}/{VIDEO_NAME}.mp4')
    else:
        os.system(f'ffmpeg -i {video_path} -vf "subtitles={RESULT_PATH}/{VIDEO_NAME}/{VIDEO_NAME}_zh.srt" {RESULT_PATH}/{VIDEO_NAME}/{VIDEO_NAME}.mp4')

