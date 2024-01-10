import os
import openai
import json
import shutil
import logging
import time
import re
from colorama import init, Fore, Back, Style
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

start_time = time.time()

load_dotenv()
init() # Colorama   

openai.api_key = os.getenv("OPENAI_API_KEY")
sony_directory = os.getenv("SONY_LOCATION")
other_audio_directory = os.getenv("OTHER_AUDIO_LOCATION")
icloud_audio_location = os.getenv("ICLOUD_LOCATION")
audiobackup_directory = os.getenv("COPY_AUDIO_TO_LOCATION")
textcreation_directory = os.getenv("COPY_TEXT_TO_LOCATION")
AudioSegment.converter = os.getenv("FFMPEG_LOCATION")
temp_directory = os.getenv("TEMP_LOCATION")
summary_time = (int(os.getenv("SUMMARY_TIME")) * 1000) # Seconds
segment_time = (int(os.getenv("SEGMENT_TIME")) * 1000) # Seconds

def check_sony_file_directory():
    if os.path.exists(sony_directory):
        return(True)
    else:
        return(False)

def check_icloud_file_directory():
    dir = os.path.expanduser(icloud_audio_location)
    if os.path.exists(dir):
        return(True)
    else:
        return(False)

def get_files_from_sony():
    return(os.listdir(sony_directory))

def get_files_from_icloud():
    dir = os.path.expanduser(icloud_audio_location)
    return(os.listdir(dir))

def create_temp_directory():
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

def summary_text_for_filename(summarytext):
    command = """
    You will be given notes that need to be summarized into 200 characters or less. 
    Do not include any date information in the summary. 
    This will need to make sense to a reader with the content produced. Here are the notes: 
    """
    
    contenttosend = command + summarytext
    
    system_content = """
    You are asummarizing notes to provide meaniingful content for naming a file.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",   
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": contenttosend}, 
         ]
    )
    print(f"    Summary Title: {str(response['choices'][0]['message']['content'])}")
    return(str(response["choices"][0]["message"]["content"]))

def break_audio_into_segments(audio, segment_length):
    segments = []
    for start_time in range(0, len(audio), segment_length):
        end_time = start_time + segment_length
        segment = audio[start_time:end_time]
        segments.append(segment)
    print(Fore.GREEN + Style.BRIGHT + f"    Number of Segments: {len(segments)}" + Style.RESET_ALL)
    return segments

def get_text_from_segment(segment):
    file = temp_directory + "/segmenttemp.mp3"
    if os.path.exists(file):
        os.remove(file)
    segment.export("./" + file, format="mp3")
    segment_file = open(file, "rb")
    response = openai.Audio.transcribe(
        model = "whisper-1", 
        file = segment_file,
        response_format = "json",
        language = "en",
    )
    if os.path.exists(file):
        os.remove(file)
    return(response["text"])

def create_audio_text(directory, file_name):
    create_temp_directory()
    dir = os.path.expanduser(directory)
    audio_file = AudioSegment.from_file(os.path.join(dir, file_name))
    segments = break_audio_into_segments(audio_file, segment_time)
    response_text = ""
    for i, segment in enumerate(segments):
        print(Fore.LIGHTGREEN_EX + f"    Processing segment: {i+1}" + Style.RESET_ALL)
        response_text += get_text_from_segment(segment)
    return(response_text)

def get_summary_for_file(directory, file_name):
    """
    Produces summary text from the begining of the audio file to provide more context to the file name
    The file text is limited to 200 characters
    """
    create_temp_directory()
    dir = os.path.expanduser(directory)
    filelocation = os.path.join(dir, file_name)
    audio_file = AudioSegment.from_file(filelocation)
    duration = ((len(audio_file) / 1000.0) / 60) # Minutes
    print(f"    Audio File Length: {duration}")
    segment = audio_file[0:summary_time]
    # Temp File
    file = temp_directory + "/tempsummary.mp3"
    if os.path.exists(file):
        os.remove(file)
    segment.export("./" + file, format="mp3")
    summary_file = open(file, "rb")
    # Transcribe the first 15 seconds
    response = openai.Audio.transcribe(
        model = "whisper-1", 
        file = summary_file,
        response_format = "json",
        language = "en",
    )
    summary_text = str(summary_text_for_filename(response["text"]))
    if len(summary_text) > 200:
        return(summary_text[0:200])
    else:
        return(summary_text)

def check_move_audio_file_directory():
    if not os.path.exists(audiobackup_directory):
        os.makedirs(audiobackup_directory)

def move_audio_file(directory, file_name):
    check_move_audio_file_directory()
    dir = os.path.expanduser(directory)
    orig_file_path = os.path.join(dir, file_name)
    dest_file_path = os.path.join(audiobackup_directory, file_name)
    shutil.copy(orig_file_path, dest_file_path)

def check_move_text_file_directory():
    if not os.path.exists(textcreation_directory):
        os.makedirs(textcreation_directory)

def delete_audio_file(directory, file_name):
    dir = os.path.expanduser(directory)
    os.remove(os.path.join(dir, file_name))

def create_text_file(audio_text, summary_text, file_name):
    check_move_text_file_directory()
    file_path = os.path.join(textcreation_directory, file_name)

    # Determine where this file should be placed in directory based on intro summary

    with open(file_path, "w") as text_file:
        text_file.write("### Summary")
        text_file.write("\n")
        text_file.write(summary_text)
        text_file.write("\n\n")
        text_file.write("### Audio Text")
        text_file.write("\n")
        text_file.write(audio_text)
        text_file.write("\n")
        text_file.close()

def parse_audio_file(audio_file_name):
    """
    Parse the audio file name to get the date and time into components
    """
    # Check The Pattern for Sony files
    # YYMMDD_HHMM.mp3
    patternmp3 = r"\d{6}_\d{4}\.mp3"
    patternm4a = r"\d{6}_\d{4}\.m4a"
    if re.match(patternmp3, audio_file_name):
        date, time = audio_file_name.removesuffix(".mp3").split("_")
        year = date[0:2]
        month = date[2:4]
        day = date[4:6]
        hour = time[0:2]
        minute = time[2:4]
        type = "mp3"
        return date, time, year, month, day, hour, minute, type
    elif re.match(patternm4a, audio_file_name):
        date, time = audio_file_name.removesuffix(".m4a").split("_")
        year = date[0:2]
        month = date[2:4]
        day = date[4:6]
        hour = time[0:2]
        minute = time[2:4]
        type = "m4a"
        return date, time, year, month, day, hour, minute, type    
    else:
        raise ValueError("File Cannot be Parsed")

def main_processing():
    """
    The orchistation of the processing of the files
    """
    sony_dir_present = False
    icloud_dir_present = False
    # Checking for directory of Sony created files 
    if not check_sony_file_directory():
        print(Fore.WHITE + Back.RED + Style.BRIGHT + "Sony directory not present to process" + Style.RESET_ALL)
    else:
        sony_dir_present = True
    
    if not check_icloud_file_directory():
        print(Fore.WHITE + Back.RED + Style.BRIGHT + "iCloud directory not present to process" + Style.RESET_ALL)
    else:
        icloud_dir_present = True

    file_list_Sony = []
    file_list_iCloud = []
    # Getting files from Sony
    if sony_dir_present:
        file_list_Sony = get_files_from_sony()

    # Getting files from iCloud
    if icloud_dir_present:
        file_list_iCloud = get_files_from_icloud()

    print(Fore.GREEN + "Number of Sony files: " + Fore.YELLOW +f"{len(file_list_Sony)}" + Style.RESET_ALL)
    print(Fore.GREEN + "Number of iCloud files: " + Fore.YELLOW +f"{len(file_list_iCloud)}" + Style.RESET_ALL)
    
    if sony_dir_present:
        # Loop through Sony files
        for file_name in file_list_Sony:
            # Sony File Name Handling
            try:
                date, time, year, month, day, hour, minute, type = parse_audio_file(str(file_name))
                print(Fore.GREEN + "Processing Sony file: " + Fore.YELLOW + f"{file_name}" + Style.RESET_ALL)

                # Changed to allow spaces for better readablity in the Obsidian application
                text_file_name = date + " " + time + " " + str(get_summary_for_file(sony_directory, file_name)) + ".md"

                # Generate Audio Text
                audio_text = create_audio_text(sony_directory, file_name)

                # Summary Text from AI
                summary_text = "This is summary Text from AI"        
                # Todos or Actions from Text
                # action_items[] = []
                
                # Produce Text File
                create_text_file(audio_text, summary_text, text_file_name)

                # Move Audio File
                move_audio_file(sony_directory, file_name)

                # Delete Audio File
                delete_audio_file(sony_directory, file_name)
                
                # Final Processing Message
                print(Fore.GREEN + Style.BRIGHT + f"    {file_name} processed" + Style.RESET_ALL)

            except ValueError:
                print(Fore.WHITE + Back.RED + Style.BRIGHT + f"Error parsing file name: {file_name}" + Style.RESET_ALL)    

    if icloud_dir_present:
        # Loop through iCloud files
        for file_name in file_list_iCloud:
            #iCloud File Handling
            try:
                date, time, year, month, day, hour, minute, type = parse_audio_file(str(file_name))
                print(Fore.GREEN + "Processing iCloud file: " + Fore.YELLOW + f"{file_name}" + Style.RESET_ALL)

                # Changed to allow spaces for better readablity in the Obsidian application
                text_file_name = date + " " + time + " " + str(get_summary_for_file(icloud_audio_location,file_name)) + ".md"

                # Generate Audio Text
                audio_text = create_audio_text(icloud_audio_location, file_name)

                # Summary Text from AI
                summary_text = "This is summary Text from AI" 

                # Todos or Actions from Text
                # action_items[] = []
                
                # Produce Text File
                create_text_file(audio_text, summary_text, text_file_name)

                # Move Audio File
                move_audio_file(icloud_audio_location, file_name)       

                # Delete Audio File
                delete_audio_file(icloud_audio_location, file_name)      

                # Final Processing Message
                print(Fore.GREEN + Style.BRIGHT + f"    {file_name} processed" + Style.RESET_ALL)                   

            except ValueError:
                print(Fore.WHITE + Back.RED + Style.BRIGHT + f"Error parsing file name: {file_name}" + Style.RESET_ALL)
        
if (True):
    main_processing()
else:
    print("Not handling files at this time")