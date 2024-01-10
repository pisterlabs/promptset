import google.auth
import openai, base64, os, cv2, json, io, re, whisper, time
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
import concurrent.futures
from googletrans import Translator
from pydub import AudioSegment
from wrapt_timeout_decorator import timeout
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from isodate import parse_duration
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
from config import settings
import variables 


def execute_function_wrapper(func, arg):
    '''This function takes a function name and it's argument and executes the said function.'''
    # Execute the provided function with its arguments
    result = func(arg)
    return result


#Functions to extract youtube vidoe content, subtitles and even download videos as well.
def search_videos_keyword(api_key, query, max_results=5):
    '''This function extracts the details a list of youtube videos given certain keywords.'''
    
    # Set up YouTube API service
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Search for videos based on keywords
    request = youtube.search().list(
        part='snippet',
        q=query,
        type='video',
        maxResults=max_results)

    response = request.execute() #Executes the query

    if 'items' in response:
        videos = response['items']
        return videos
    else:
        print("No videos found.")


def search_videos_channel(api_key, channel_id, max_results=50):
    '''This function takes in the YouTube API key, a channel ID and results threshold
    and extract a limited number of video metadata with respected to the provided threshold.'''

    # Set up YouTube API service
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # Get the playlist ID of the uploads playlist for the channel
    response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Get the video details from the uploads playlist
    videos = []
    next_page_token = None
    while True:
        playlist_items = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=max_results,
            pageToken=next_page_token,
        ).execute()

        videos.extend(playlist_items["items"])
        next_page_token = playlist_items.get("nextPageToken")

        if (not next_page_token) | (len(videos) >= max_results):
            break
    return videos[:max_results]


def get_youtube_video_info(api_key, video_id):
    '''This function extracts the details of a youtube video using its video ID'''
    
    # Set up YouTube API service
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Get video details
    request = youtube.videos().list(part='snippet,contentDetails,statistics', id=video_id)
    response = request.execute()

    #Extract the important content from the response
    if 'items' in response:
        video_info = response['items'][0]
        return video_info
    else:
        print("Video not found.")
        return None


def convert_duration_to_seconds(duration):
    '''This function converts the video duration extracted from youtuve details to seconds'''

    # Parse the ISO 8601 duration format
    duration_obj = parse_duration(duration)

    # Calculate the total duration in seconds
    total_seconds = duration_obj.total_seconds()

    return int(total_seconds)


def get_video_transcript(video_ids, languages):
    '''This function extracts the subtitle of a youtube video using its video ID'''

    transcript = YouTubeTranscriptApi.get_transcripts(video_ids, languages=languages)
    transcript = transcript[0][video_ids[0]]
    return transcript
    

def save_transcript_to_file(transcript, output_file):
    '''This functions saves the subtitle extracted from the chosen video.'''

    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in transcript:
            file.write(f"{entry['start']} - {entry['start'] + entry['duration']}\n")
            file.write(f"{entry['text']}\n\n")


def speech_speed(video_transcript, subtitle_language):
    '''This function takes in the video transcript and calculates the speed of the speech in the video.'''

    translator = Translator()
    combined_duration = 0 #variable to store the total number of seconds of speech in the video
    number_of_words = 0 #variable to store the total number of words in the video
    speed_categories = {'Slow Speech':[0,110],'Normal Speech':[110,150],'Fast Speech':[150,200]}

    music = 'Music'
    if subtitle_language != 'en':
        translated_text = translator.translate(music, src='en', dest=subtitle_language)
        music = (translated_text.text).replace('\n',' ')

    for text in video_transcript:
        if (text['text'] != f"[{music}]") & (text['text'] != f"[Music]"): #Excludes the parts of the script that only contains just music
            combined_duration += int(text['duration'])
            number_of_words += int(len(text['text'].split(' ')))

    #calculates the words per minute 
    words_per_minute = round(number_of_words/(combined_duration/60))
    for categ in list(speed_categories.keys()):
        #using the calculated words per minute, the speech speed category is determined.
        if (words_per_minute >= speed_categories[categ][0]) & (words_per_minute < speed_categories[categ][1]):
            audio_speed = categ
    #print(combined_duration, number_of_words, f"{words_per_minute} WPM", audio_speed)
    return combined_duration, words_per_minute, audio_speed


def subt_sing_translator(input_set):
    '''This function takes in a part of an extracted subtitle and translates it using Google's API.'''
    
    translator = Translator()
    
    # Translates the chunk of the subtitle using the source languages.
    index = input_set[0]
    text = input_set[1]
    source_language = input_set[2]

    try:
        translated_text = translator.translate(text, src=source_language, dest='en')
        translated_subt = (translated_text.text).replace('\n',' ')
    except Exception as e:
        # print(f"Error: {e}")
        # print(index, text)
        translated_subt = ''

    return (index, translated_subt)


def subt_set_translator(sublist):
    '''This function takes in a sublists containing parts of an extracted sutitle,
    and then translates it in a parallel manner (Multithreaded).'''

    translated_dict = {}
    #Multithreading (I/O bound)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        arguments = sublist
        futures = [executor.submit(subt_sing_translator, arg) for arg in arguments]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result[1] != '':
                    translated_dict[int(result[0])] = result[1]
            except Exception as e:
                # print(f"Error: {e}")
                continue
    return translated_dict


def combine_transcript_translate(transcript, source_language):
    '''This processes the extracted subtitle, translates (to English in a parallel manner, 
    if text isn't already in English) and combines all its texts into one long string.'''

    string = '' #Declares an initial empty string
    
    if source_language == 'en':
        # print("English")
        #Loops through the extracted transcript to compile it for further processing
        for subt in transcript:
            if subt['text'] != '[Music]':
                string = string+f" {subt['text']}"
        return string
    else:
        # print("Not English")
        #The parts of the subtitle are enumerated to be processed in paralled
        translator = Translator()

        #Translates the word 'Music' using the video language so as to omit segments containing [Music] in the subtitles
        translated_text = translator.translate('Music', src='en', dest=source_language)
        music = (translated_text.text).replace('\n',' ')

        list_of_subts = [(i, transcript[i]['text'], source_language) for i in range(len(transcript)) if (transcript[i]['text'] != f"[{music}]") & (transcript[i]['text'] != f"[Music]")]
        
        #The list of texts are further divided into set of lists which are to processed in parallel
        len_of_sublists = int(round(len(list_of_subts)/4))
        sublist_of_subts = [list_of_subts[i:i+len_of_sublists] for i in range(0, len(list_of_subts), len_of_sublists)]
        
        translated_dict = {}
        #Multiprocessing (CPU bound)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            arguments = sublist_of_subts
            results = executor.map(subt_set_translator, arguments)

            for result in results:
                for key in list(result.keys()):
                    translated_dict[int(key)] = result[key]

        #The punctuated text chunks are then rearranged into a meaningful order using their original assigned index.
        ordered_keys = sorted(list(translated_dict.keys()))

        #Loops through the extracted and translated transcript to compile it for further processing
        for ordered_index in ordered_keys:
            string = string+f" {translated_dict[ordered_index]}"
        return string


def subt_sing_punctuator(part_sub):
    '''This function takes in a part of a combined subtitle and punctuates it using GPT's API.'''

    print(f"Length of text being analysed: {len(part_sub[1])}")
    try:
        combined_subt_punct = gpt_punctuator(part_sub[1])
        # print(combined_subt_punct[:10])
        return (part_sub[0], combined_subt_punct)
    except:
        # print("No response")
        return (part_sub[0], '')


def subt_set_punctuator(sublist):
    '''This function takes in a sublists containing parts of a combined sutitle,
    and then processes it in a parallel manner (Multithreaded).'''

    punctuated_dict = {}
    #Multithreading (I/O bound)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        arguments = sublist
        futures = [executor.submit(subt_sing_punctuator, arg) for arg in arguments]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result[1] != '':
                    punctuated_dict[int(result[0])] = result[1]
            except:
                continue
    return punctuated_dict


def subtitle_processing(combined_subt):
    '''This function takes the combined raw subtitle and punctuates it using GPT in a parallel
    manner (Multiprocessor).'''

    def split_and_enumerate_1(combined_subt, trunc_threshold):
        '''This function split the combined subtitle and enumerates each split
        to be processed in parallel, using the word-based method.'''

        split_subtitle = combined_subt.split(' ')
        print(f"Number of words: {len(split_subtitle)}\nTruncation threshold: {trunc_threshold}")
        subtitle_list = []
        combined_words, count, combined_count = '', 0, 0

        #This splits the entire combined subititle using the threshold calculated from the available cores
        for word in split_subtitle:
            combined_words = combined_words + f" {word}"
            count += len(word)
            if count >= trunc_threshold:
                #The split texts are appended to a a list with assigned indices
                subtitle_list.append((combined_count, combined_words)) 
                combined_words, count, combined_count = '', 0, combined_count+1
        subtitle_list.append((combined_count, combined_words))
        return subtitle_list

    def split_and_enumerate_2(combined_subt, trunc_threshold):
        '''This function split the combined subtitle and enumerates each split
        to be processed in parallel, using the character-based method.'''

        start_index, end_index, index_list = 0, trunc_threshold, []
        len_of_string = len(combined_subt)
        while end_index < len_of_string:
            #This loop demarcates the length of the combined string into intervals for indexing
            index_list.append((start_index, end_index))
            start_index = end_index
            end_index = end_index + trunc_threshold
        index_list.append((start_index, len(combined_subt)))

        subtitle_list = [(i, combined_subt[index_list[i][0]:index_list[i][1]]) for i in range(len(index_list))]
        return subtitle_list

    # Preprocesses the subtitle, so that GPT can process it without trucnating it.
    len_of_combined_subt = len(combined_subt)
    num_of_chunk_in_sublist = 2
    num_of_cores = 4
    trunc_threshold = round(len_of_combined_subt/(num_of_cores*num_of_chunk_in_sublist)) #Uses the number of available cores (4) to split the text for quick processing

    #Calls the function to split and enumerate the combined subtitle
    subtitle_list = split_and_enumerate_2(combined_subt, trunc_threshold)

    #The list of texts are further divided into set of lists which are to processed in parallel 
    len_of_sublists = int(round(len(subtitle_list)/num_of_cores))
    sublist_of_subts = [subtitle_list[i:i+len_of_sublists] for i in range(0, len(subtitle_list), len_of_sublists)]
    print(f"Number of sublists: {len(sublist_of_subts)}")

    subt_dict = {}
    #Multiprocessing (CPU bound)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        arguments = sublist_of_subts
        results = executor.map(subt_set_punctuator, arguments)

        for result in results:
            for key in list(result.keys()):
                subt_dict[int(key)] = result[key]

    #The punctuated text chunks are then rearranged into a meaningful order using their original assigned index.
    ordered_keys = sorted(list(subt_dict.keys()))
    punct_subt_list = [subt_dict[key] for key in ordered_keys]  
    
    #The list of punctuated subtitles are combined once more to form a whole.
    for i in range(len(punct_subt_list)):
        if i == 0:
            final_combined_punct_subt = punct_subt_list[i]
        else:
            final_combined_punct_subt = final_combined_punct_subt + f" {punct_subt_list[i]}"

    gpt_threshold = 13000
    #The punctuated subtitles is then truncated to fit GPT's token limit for processing.
    trunc_string = final_combined_punct_subt[:gpt_threshold]
    print(f"Length: of truncated punctuated subtitle: {len(trunc_string)}")
    return final_combined_punct_subt, trunc_string


def download_youtube_video(video_url, output_path='.'):
    '''This function downloads a given youtube video using its video url.'''

    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the highest resolution stream
        video_stream = yt.streams.get_highest_resolution()

        # Download the video
        video_stream.download(output_path)
        print(f"Video downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Error: {e}")



#Functions for executing text analysis and processor (classification, summarization, topic modelling).
def gpt_punctuator(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''

    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant. Your job is to punctuate a given text and output only the resulting punctuated text without omiting a single word."}]

    #Creates the prompt to punctuate the subtitle extracted from the given video
    prompt_1 = f"{information}"
    prompt_2 = "Please properly punctuate the given text (without omitting a single word) and output only the resulting punctuated text. Please do not omit a single word from the original text."

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_1},)
    messages.append({"role": "user", "content": prompt_2},)

    #GPT model is triggered and response is generated.
    chat = openai_obj.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        temperature=0.0,
        #timeout=5
    ) 

    #Response is extracted
    response = chat.choices[0].message.content
    return (response)


def gpt_categorizer(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''
    
    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant. Given a text to analyze, you're to only respond with 'Basic','Medium', or 'Advanced'."}]

    #Creates the prompt to check for the most similar column
    prompt_1 = f"{information}"
    prompt_2 = "Given the text which is a transcript of a language tutorial video, which category of difficulty (Basic, Medium and Advanced) best describes what is being taught? Output only the category and nothing else."

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_1},)
    messages.append({"role": "user", "content": prompt_2},)

    #GPT model is triggered and response is generated.
    chat = openai_obj.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        temperature=0.0,
        #timeout=5
    ) 

    #Response is extracted
    response = chat.choices[0].message.content
    return (response)


def gpt_summarizer(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''
    
    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant. Given a text to analyze, you're to summarize the content in a few sentences."}]

    #Creates the prompt to check for the most similar column
    prompt_1 = f"{information}"
    prompt_2 = "Given the text which is a transcript of a language tutorial video, please summarize the content in 5 to 10 sentences."

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_1},)
    messages.append({"role": "user", "content": prompt_2},)

    #GPT model is triggered and response is generated.
    chat = openai_obj.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        temperature=0.0,
        #timeout=5
    ) 

    #Response is extracted
    response = chat.choices[0].message.content
    return (response)


def gpt_topicmodeller(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''
    
    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant. Given a text to analyze, you're to generate a single topic that best represent the contents within, and output only this topic with no additional write up."}]

    #Creates the prompt to check for the most similar column
    prompt_1 = f"{information}"
    prompt_2 = "Given the text which is a transcript of a language tutorial video, please generate a single topic that describes the content being taught. Output only this topic and nothing else (no additional write up)."

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_1},)
    messages.append({"role": "user", "content": prompt_2},)

    #GPT model is triggered and response is generated.
    chat = openai_obj.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        temperature=0.0,
        #timeout=5
    ) 

    #Response is extracted
    response = chat.choices[0].message.content
    return (response)


def gpt_qualitycheck(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''
    
    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant. Given a text to analyze, you're to respond with only 'Poorly articulated','Moderately articulated' or 'Very articulated'."}]

    #Creates the prompt to check for the most similar column
    prompt_1 = f"{information}"
    prompt_2 = "Given the text which is a transcript of a language tutorial video, is the content 'Poorly articulated', 'Moderately articulated', or 'Very articulated'? Output only the category and nothing else."

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_1},)
    messages.append({"role": "user", "content": prompt_2},)

    #GPT model is triggered and response is generated.
    chat = openai_obj.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        temperature=0.0,
        #timeout=5
    ) 

    #Response is extracted
    response = chat.choices[0].message.content
    return (response)


def gpt_vocabularycheck(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''
    
    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant. Given a text to analyze, you're to respond with only 'Basic','Intermediate' or 'Advanced'."}]

    #Creates the prompt to check for the most similar column
    prompt_1 = f"{information}"
    prompt_2 = "Given the text which is a transcript of a language tutorial video, is the vocabulary level 'Basic', 'Intermediate', or 'Advanced'? Output only the category and nothing else."

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_1},)
    messages.append({"role": "user", "content": prompt_2},)

    #GPT model is triggered and response is generated.
    chat = openai_obj.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        temperature=0.0,
        #timeout=5
    ) 

    #Response is extracted
    response = chat.choices[0].message.content
    return (response)


def gpt_sentenceconstruct(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''
    
    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant. Given a text to analyze, you're to respond with only 'Basic','Intermediate' or 'Advanced'."}]

    #Creates the prompt to check for the most similar column
    prompt_1 = f"{information}"
    prompt_2 = "Given the text which is a transcript of a language tutorial video, is the sentence structure 'Basic', 'Intermediate', or 'Advanced'? Output only the category and nothing else."

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_1},)
    messages.append({"role": "user", "content": prompt_2},)

    #GPT model is triggered and response is generated.
    chat = openai_obj.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        temperature=0.0,
        #timeout=5
    ) 

    #Response is extracted
    response = chat.choices[0].message.content
    return (response)


def gpt_dialogue(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''
    
    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant. Given a text to analyze, you're to respond with only 'Present', or 'Not Present'."}]

    #Creates the prompt to check for the most similar column
    prompt_1 = f"{information}"
    prompt_2 = "Given the text which is a transcript of a language tutorial video, is there any dialogue present? Output only the response and nothing else."

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_1},)
    messages.append({"role": "user", "content": prompt_2},)

    #GPT model is triggered and response is generated.
    chat = openai_obj.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        temperature=0.0,
        #timeout=5
    ) 

    #Response is extracted
    response = chat.choices[0].message.content
    return (response)


def text_sing_analyzer(input_set):
    '''This function takes in a category and a truncated string and conducts a particular
    type of analysis based on the category'''

    #Extracts the contents of the input
    category = input_set[0]
    trunc_string = input_set[1]

    #Uses this dictionary to map a given category to it's corresponding GPT function.
    textanalysis_dict = {'category':gpt_categorizer,'summary':gpt_summarizer,'topic':gpt_topicmodeller,
                         'quality':gpt_qualitycheck,'vocabulary':gpt_vocabularycheck,
                         'sentence_construct':gpt_sentenceconstruct,'dialogue':gpt_dialogue}
    
    print(f"Category of Text Anlysis: {category}.")
    try:
        gpt_response = execute_function_wrapper(textanalysis_dict[category], trunc_string)
    except:
        gpt_response = ''
    return (category, gpt_response)


def text_set_analyzer(sublist):
    '''This function takes in a sublist of categories of text analysis
    and then processes it in a parallel manner (Multithreaded).'''

    test_analysis_dict = {}
    #Multithreading (I/O bound)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        arguments = sublist
        futures = [executor.submit(text_sing_analyzer, arg) for arg in arguments]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                test_analysis_dict[result[0]] = result[1]
            except:
                continue
    return test_analysis_dict



#Functions for extracting the audio from the downloaded video and analyzing this audio.
def extract_audio_from_video(video_path, audio_path):
    '''This function extracts the audio file from the downloaded youtube video.'''

    video_clip = VideoFileClip(video_path) #Loads the downloaded video
    audio_clip = video_clip.audio #Extracts the audio from the video
    audio_clip.write_audiofile(audio_path, fps=44100)  # Set the desired sample rate


def download_audio(video_id, output_path='audio_files'):
    '''This fucntion downloads the video audio file using the video ID and returns the 
    file paths'''

    try:
        # Construct the YouTube video URL
        video_url = f'https://www.youtube.com/watch?v={video_id}'

        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the highest quality audio stream
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()

        # Remove invalid characters from the title to create a valid filename
        video_title = yt.title
        valid_filename = "".join(c for c in video_title if c.isalnum() or c in (' ', '.', '_'))

        # Set the output path (default: 'downloads')
        audio_stream.download(output_path, filename=f"{valid_filename}.mp4")

        # Get the downloaded audio file path
        mp4_path = f"{output_path}/{valid_filename}.mp4"

        # Convert the downloaded audio to MP3
        audio_clip = AudioFileClip(mp4_path)
        wav_path = (f"{output_path}/{valid_filename}.wav")
        audio_clip.write_audiofile(wav_path, fps=20000)

        print(f"Audio downloaded and converted to MP3 successfully.")
        return mp4_path, wav_path
    except Exception as e:
        print(f"Error: {e}")
        return None


def analyze_audio_speed(audio_path):
    '''This function analyses the speed of the audio file.'''

    try:
        y, sr = librosa.load(audio_path) #Loads the extracted and stored audio

        # Compute the tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print(f'Tempo: {tempo} BPM')
        return tempo
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def analyze_sing_audio_speed(input_set):
    '''This function takes in a set containing audio segment interval, audio file path and 
    the set index, analyzes the audio speed of the segment and returns this value along with
    its index.'''
    # Extract the segment, audio time interval and the languages
    index = input_set[0]
    time_interval = input_set[1]
    audio = AudioSegment.from_file(input_set[2])
    segment = audio[time_interval[0]:time_interval[1]]

    # Save the segment to a temporary file
    temp_file_path = f"audio_files/temp_segment_{index}.wav"
    segment.export(temp_file_path, format="wav")

    try:
        y, sr = librosa.load(temp_file_path) #Loads the extracted and stored audio

        # Compute the tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception as e:
        print(f"Error: {e}")
        tempo = None
    
    os.remove(temp_file_path) #Deletes the audio segment after processing to free up space
    return tempo


def analyze_set_audio_speed(audio_path):
    '''This function analyses the speed of the audio file.'''

    try:
        # Load the entire audio file
        audio = AudioSegment.from_file(audio_path)

        segment_duration_ms = len(audio)/4

        # Calculate the number of segments
        num_segments = len(audio) // segment_duration_ms
        print(int(num_segments))

        list_of_segments = []
        for i in range(int(num_segments)):
            # Calculate start and end time for each segment
            start_time = i * segment_duration_ms
            end_time = (i + 1) * (segment_duration_ms/1.5)
            list_of_segments.append((i, [start_time, end_time], audio_path,))

        bpm = []
        #Multiprocessing (CPU bound)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            arguments = list_of_segments
            results = executor.map(analyze_sing_audio_speed, arguments)

            for result in results:
                if result != None:
                    bpm.append(result)

        average_tempo = round(sum(bpm)/len(bpm))
        print(f'Tempo: {average_tempo} BPM')
        return average_tempo
    except Exception as e:
        print(f"Error: {e}")
        return None


def audiolang_sing_processor_google(input_set):
    '''This function takes in the set of input necesasry to process the audio segment,
    in order to execute a parallel process'''
    
    count_overall, count_transcribed, count_firstlang, count_secondlang = 1, 0, 0, 0
    recognizer = sr.Recognizer()
    
    # Extract the segment, audio time interval and the languages
    index = input_set[0]
    time_interval = input_set[1]
    audio = AudioSegment.from_file(input_set[2])
    language_list = input_set[3]
    segment = audio[time_interval[0]:time_interval[1]]

    # Save the segment to a temporary file
    temp_file_path = f"audio_files/temp_segment_{index}.wav"
    segment.export(temp_file_path, format="wav")

    try:
        # Transcribe the segment while trying the first language
        with sr.AudioFile(temp_file_path) as audio_file:
            audio_data = recognizer.record(audio_file)
            text = recognizer.recognize_google(audio_data, language=language_list[0])
            # print(f"Segment {index + 1} Transcription:", text)
            count_transcribed = 1
            count_firstlang = 1
    except sr.UnknownValueError:
        try:
            # Transcribe the segment while trying the second language
            with sr.AudioFile(temp_file_path) as audio_file:
                audio_data = recognizer.record(audio_file)
                text = recognizer.recognize_google(audio_data, language=language_list[1])
                # print(f"Segment {index + 1} Transcription:", text)
                count_transcribed = 1
                count_secondlang = 1
        except sr.UnknownValueError:
            # print(f"Segment {index + 1} - Speech Recognition could not understand audio")
            pass
    except sr.RequestError as e:
        # print(f"Segment {index + 1} - Could not request results from Google Speech Recognition service; {e}")
        pass

    os.remove(temp_file_path) #Deletes the audio segment after processing to free up space
    return (count_overall, count_transcribed, count_firstlang, count_secondlang)
    

def audiolang_set_processor_google(sublist):
    '''This function takes in a sublist of audio segment details and processes
    it in a parallel'''

    count_overall, count_transcribed, count_firstlang, count_secondlang = [], [], [], []
    #Multithreading (I/O bound)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        arguments = sublist
        futures = [executor.submit(audiolang_sing_processor_google, arg) for arg in arguments]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  # Get the result, may raise an exception
                count_overall.append(result[0])
                count_transcribed.append(result[1])
                count_firstlang.append(result[2])
                count_secondlang.append(result[3])
            except:
                count_overall.append(1)
                count_transcribed.append(result[0])
                count_firstlang.append(result[0])
                count_secondlang.append(result[0])
    count_overall, count_transcribed = sum(count_overall), sum(count_transcribed)
    count_firstlang, count_secondlang = sum(count_firstlang), sum(count_secondlang)
    #print(count_overall, count_transcribed, count_firstlang, count_secondlang)
    return (count_overall, count_transcribed, count_firstlang, count_secondlang)


def analyze_audio_languages_google(audio_path, first_language, second_language, segment_duration_ms=4000):
    '''This fucntion downloads the video audio file using the video ID and calculated the
    audio BPM (Beats per minute).'''

    language_isocode = variables.language_isocode
    try:
        language_list = []
        for language in [first_language.lower(), second_language.lower()]:
            language_list.append(language_isocode[language])
        print(language_list)

        # Load the entire audio file
        audio = AudioSegment.from_file(audio_path)

        # Calculate the number of segments
        num_segments = len(audio) // segment_duration_ms + 1
        print(num_segments)

        list_of_segments = []
        for i in range(num_segments):
            # Calculate start and end time for each segment
            start_time = i * segment_duration_ms
            end_time = (i + 1) * segment_duration_ms
            list_of_segments.append((i, [start_time, end_time], audio_path, language_list))

        len_of_sublists = int(round(len(list_of_segments)/4))
        segments_sublist = [list_of_segments[i:i+len_of_sublists] for i in range(0, len(list_of_segments), len_of_sublists)]

        count_overall, count_transcribed, count_firstlang, count_secondlang = [], [], [], []
        #Multiprocessing (CPU bound)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            arguments = segments_sublist
            results = executor.map(audiolang_set_processor_google, arguments)

            for result in results:
                count_overall.append(result[0])
                count_transcribed.append(result[1])
                count_firstlang.append(result[2])
                count_secondlang.append(result[3])

        count_overall, count_transcribed = sum(count_overall), sum(count_transcribed)
        count_firstlang, count_secondlang = sum(count_firstlang), sum(count_secondlang)
        print(count_overall, count_transcribed, count_firstlang, count_secondlang)

        #Computes teh percentage distribution of the languages using the extracted information
        percentage_transcribed = round((count_transcribed/count_overall)*100)
        percentage_firstlang = round((count_firstlang/count_transcribed)*100)
        percentage_secondlang = 100-percentage_firstlang
        # print(f"Percentage transcribed: {percentage_transcribed}%, {first_language}: {percentage_firstlang}%, {second_language}: {percentage_secondlang}%")
        return percentage_transcribed, percentage_firstlang, percentage_secondlang
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def openai_whisper_api(path):
    '''This function takes in a file path and loads the audio file at the end of this
    file path onto the openai_whisper_api for transcription.'''

    model = whisper.load_model("base")
    result = model.transcribe(path)
    #print(result['text'])
    language = result['language']
    return language


def audiolang_sing_processor_openai(input_set):
    '''This function takes in the set of input necesasry to process the audio segment,
    in order to execute a parallel process.'''
    
    count_overall, count_transcribed, language = 1, 0, None
    
    # Extract the segment, audio time interval and the languages
    index = input_set[0]
    time_interval = input_set[1]
    audio = AudioSegment.from_file(input_set[2])
    segment = audio[time_interval[0]:time_interval[1]]

    # Save the segment to a temporary file
    temp_file_path = f"./audio_files/temp_segment_{index}.wav"
    segment.export(temp_file_path, format="wav")

    try:
        language = openai_whisper_api(temp_file_path)
        count_transcribed = 1
    except Exception as e:
        print(f"Segment {index + 1} - Speech Recognition could not understand audio: {e}")

    os.remove(temp_file_path) #Deletes the audio segment after processing to free up space
    return (count_overall, count_transcribed, language)


def audiolang_set_processor_openai(sublist):
    '''This function takes in a sublist of audio segment details and processes
    it in a parallel'''

    count_overall, count_transcribed, lang_dict = [], [], {}
    #Multithreading (I/O bound)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        arguments = sublist
        futures = [executor.submit(audiolang_sing_processor_openai, arg) for arg in arguments]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Get the result, may raise an exception
            count_overall.append(result[0])
            count_transcribed.append(result[1])
            if result[2] != None:
                if result[2] in list(lang_dict.keys()):
                    lang_dict[str(result[2])].append(1)
                else:
                    lang_dict[str(result[2])] = [1]
    count_overall, count_transcribed = sum(count_overall), sum(count_transcribed)
    langs_dict = {str(key):sum(lang_dict[key]) for key in list(lang_dict.keys())}
    #print(count_overall, count_transcribed, langs_dict)
    return (count_overall, count_transcribed, langs_dict)


def analyze_audio_languages_openai(audio_path, segment_duration_ms=4000):
    '''This fucntion downloads the video audio file using the video ID and calculated the
    audio BPM (Beats per minute).'''

    language_isocode = {'english':'en-US', 'italian':'it-IT', 'french':'fr-FR'}
    try:
        # Load the entire audio file
        audio = AudioSegment.from_file(audio_path)

        # Calculate the number of segments
        num_segments = len(audio) // segment_duration_ms + 1
        print(num_segments)

        list_of_segments = []
        for i in range(num_segments):
            # Calculate start and end time for each segment
            start_time = i * segment_duration_ms
            end_time = (i + 1) * segment_duration_ms
            list_of_segments.append((i, [start_time, end_time], audio_path))

        len_of_sublists = int(round(len(list_of_segments)/4))
        segments_sublist = [list_of_segments[i:i+len_of_sublists] for i in range(0, len(list_of_segments), len_of_sublists)]

        count_overall, count_transcribed, language_dict = [], [], {}
        #Multiprocessing (CPU bound)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            arguments = segments_sublist
            results = executor.map(audiolang_set_processor_openai, arguments)

            for result in results:
                count_overall.append(result[0])
                count_transcribed.append(result[1])
                for key in list(result[2].keys()):
                    if key in (language_dict.keys()):
                        language_dict[key].append(result[2][key])
                    else:
                        language_dict[key] = [result[2][key]]

        count_overall, count_transcribed = sum(count_overall), sum(count_transcribed)
        languages_dict = {str(key):((sum(language_dict[key])/count_transcribed)*100) for key in list(language_dict.keys())}
        languages_dict = {k:f"{v}%" for k, v in sorted(languages_dict.items(), key=lambda item: item[1], reverse=True)}
        print(count_overall, count_transcribed, languages_dict)

        #Computes teh percentage distribution of the languages using the extracted information
        percentage_transcribed = round((count_transcribed/count_overall)*100)
        print(f"Percentage transcribed: {percentage_transcribed}%,\n{languages_dict}")
        return percentage_transcribed, languages_dict
    except Exception as e:
        print(f"Error: {e}")
        return None


def delete_audios(path_list):
    '''This function takes in audio file paths and deletes them from the system.'''

    for audio_path in path_list:
        os.remove(audio_path)
        print(f"Removed {audio_path} from the repository!")



#Functions to analyze the image frames extracted from the downloaded video
def extract_frames(video_path, output_folder):
    '''This function extract image frames from the downloaded youtube video.'''

    cap = cv2.VideoCapture(video_path) # Open the video file

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read and save frames
    frame_count = 0
    while True:
        if frame_count == 5:
            #Limits the number of images extracted to 5
            break
        else:
            ret, frame = cap.read()
            if not ret:
                break
            # Save the frame as an image file
            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

    # Release the video capture object
    cap.release()


def list_files_in_folder(folder_path):
    '''This function add the names of the image frames extracted from the downloaded video to a list.'''

    list_of_contents = [] #Creates an empty list to populate with the contents of the selected folder
    try:
        # Get the list of files and directories in the specified folder
        contents = os.listdir(folder_path)

        # Print the list of contents
        print(f"Contents of {folder_path}:")
        for entry in contents:
            list_of_contents.append(str(entry))
            print(entry)
        return list_of_contents
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access '{folder_path}'.")


def gpt_v_image_analyser(image_name):
    '''This function converts the extracted image frames to base64 and analyzes its content using GPT4-V'''

    import openai
    openai_obj = openai
    openai_obj.api_key = settings.openai_apikey

    # Updated file path to a JPEG image
    image_path_base = r".\output_frames\\"
    
    image_path = image_path_base + image_name

    # Read and encode the image in base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Craft the prompt for GPT
    prompt_messages = [{"role": "user",
                    "content": [{"type": "text", "text": "Does this image contain any infographics? Reply with only 'Yes' or 'No' and no added punctuations."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}]
                   }]

    # Send a request to GPT
    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "api_key": settings.openai_api,
        # "response_format": {"type": "json_object"},
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 4096,
    }

    # result = openai.ChatCompletion.create(**params)
    result = openai_obj.chat.completions.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content

