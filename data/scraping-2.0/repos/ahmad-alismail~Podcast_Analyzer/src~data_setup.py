"""prepare and download data if needed"""
import pandas as pd
import re 
import os
import yt_dlp
from pydub import AudioSegment
import openai

def clean_title(title):
    """
    Cleans a video title for use as a filename.

    Args:
        title (str): The original title to be cleaned.

    Returns:
        str: The cleaned title, safe for use as a filename.
    """
    title = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', title)
    title = re.sub(r'\|.*?\d+', '', title)
    title = title.rstrip().replace(' ', '_').replace(':', '_').replace('&','and').lower()
    title = re.sub(r'[^a-zA-Z0-9_]', '', title)
    return title

def download_audio(link, quality='64'):
    """
    Download audio from a YouTube video, convert to MP3, and save to disk.

    Args:
        link (str): YouTube video URL.
        quality (str): Desired audio quality in kbps. Default is '192'.

    Returns:
        str: Path of the saved MP3 file.
    """
    with yt_dlp.YoutubeDL({
        'format': 'bestaudio/best',
        'outtmpl': 'data/audio/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': quality,
        }],
        'ffmpeg_location': '/usr/bin/ffmpeg' 
    }) as video:
        info_dict = video.extract_info(link, download=True)
        video_id = info_dict['id']
        video_title = info_dict['title']

        title = clean_title(video_title)
        original_file_path = 'data/audio/' + video_id + '.mp3'
        new_file_path = 'data/audio/' + title + '.mp3'
        os.rename(original_file_path, new_file_path)

    return new_file_path


def transcribe_audio(audio_file_path, user_openai_key, chunk_length_min=20):
    """
    Transcribe the audio file at the given path using OpenAI's Whisper ASR system.

    Args:
        audio_file_path (str): Path to the audio file to transcribe.
        chunk_length_min (float): Length in minutes for each chunk of the audio file. Default is 45 minutes.

    Returns:
        str: The file name containing the combined transcription of all chunks of the audio file.
    """
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"No such file: '{audio_file_path}'")
    openai.api_key = user_openai_key

    source = AudioSegment.from_mp3(audio_file_path)
    whisper_fname = os.path.splitext(os.path.basename(audio_file_path))[0]
    whisper_transcript_path = f"data/{whisper_fname}.vtt"
    
    if source.duration_seconds < chunk_length_min * 60:
        with open(audio_file_path, "rb") as audio_file:
            whisper_output = openai.Audio.transcribe("whisper-1", audio_file, response_format="vtt")
        with open(whisper_transcript_path, "w") as f:
            f.write(whisper_output)
    else:
        chunk_length = chunk_length_min * 60 * 1000  # PyDub works in milliseconds
        transcriptions = []
        for i, chunk in enumerate(source[::chunk_length]):
            chunk_file_path = f"chunk_{i}.mp3"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk.export(chunk_file, format="mp3")
            with open(chunk_file_path, "rb") as chunk_file:
                chunk_transcription = openai.Audio.transcribe("whisper-1", chunk_file, response_format="vtt")
                transcriptions.append(chunk_transcription)
            os.remove(chunk_file_path)  # remove the chunk file after use

        # Combine the transcriptions of all chunks
        combined_transcription = "\n".join(transcriptions)
        with open(whisper_transcript_path, "w") as f:
            f.write(combined_transcription)

    return whisper_fname



def convert_vtt_to_csv(TRANSCRIPT_PATH, TRANSCRIPT_FNAME_VTT, TRANSCRIPT_FNAME_CSV):
    """
    This function converts a VTT transcript file to a CSV format. 

    Parameters:
    TRANSCRIPT_PATH (str): The path to the transcript files.
    TRANSCRIPT_FNAME_VTT (str): The filename of the VTT transcript.
    TRANSCRIPT_FNAME_CSV (str): The desired filename for the output CSV transcript.
    """
    with open(f"{TRANSCRIPT_PATH}/{TRANSCRIPT_FNAME_VTT}") as oldfile, open(f"{TRANSCRIPT_PATH}/{TRANSCRIPT_FNAME_CSV}", 'w') as newfile:
        old_lines = oldfile.read().split('\n')
        clean_lines = [line for line in old_lines if line not in ['', 'WEBVTT']]

        for line_idx in range(0, len(clean_lines)-1, 2):
            timestamp = clean_lines[line_idx].split('-->')[0].strip()
            # Remove milliseconds
            #timestamp = timestamp.split('.')[0]
            # Standardize timestamp format
            timestamp = "00:" + timestamp if len(timestamp.split(':')) < 3 else timestamp
            timestamp = "0" + timestamp if len(timestamp.split(':')[0]) < 2 else timestamp
            text = clean_lines[line_idx+1].rstrip()
            new_line = f"{timestamp};{text}\n"
            newfile.write(new_line)



def reorganize_transcript(df):
    """
    This function recreates the transcript dataframe, concatenating partial sentences into full ones.
    
    Parameters:
    df (pandas.DataFrame): A dataframe that contains timestamped text transcripts.
    
    Returns:
    transcript_df (pandas.DataFrame): A dataframe with full sentences for analysis purposes.
    """
    # Recreate the dataframe with full sentences
    transcript_df = pd.DataFrame(columns=df.columns)

    for idx, timestamp, text in df.itertuples():
        while text[-1] != '.':
            idx += 1
            text += df.loc[idx]['text']
        transcript_df = pd.concat([transcript_df, pd.DataFrame({'timestamp': timestamp, 'text': text}, index=[0])], ignore_index=True)

    # Remove any piece of text if it is included in previous text
    not_part_of_previous = [True]
    for i in range(1, len(transcript_df)):
        not_part_of_previous.append(transcript_df['text'][i] not in transcript_df['text'][i-1])
    transcript_df = transcript_df[not_part_of_previous] 

    return transcript_df


def prepare_transcript_for_modelling(transcript_df):
    """
    This function prepares a transcript dataframe for text summarization and topic modelling 
    
    Parameters:
    transcript_df (pandas.DataFrame): A dataframe that contains timestamped text transcripts.

    Returns:
    transcript_df_topic (pandas.DataFrame)
    """
    transcript_df['group'] = transcript_df.index // 8
    transcript_df_topic = transcript_df.groupby('group').agg({
        'timestamp': 'first',
        'text': ' '.join
    })
    
    return transcript_df_topic


def prepare_transcript_for_book_extraction(transcript_df):
    """
    This function prepares a transcript dataframe for book titles extraction 
    
    Parameters:
    transcript_df (pandas.DataFrame): A dataframe that contains timestamped text transcripts.

    Returns:
    grouped_transcript_df (pandas.DataFrame)
    """
    transcript_df['group'] = transcript_df.index // 20
    grouped_transcript_df = transcript_df.groupby('group').agg({
        'timestamp': 'first',
        'text': ' '.join,
        'is_book_related': 'any',
        'book_candidates': 'sum',
        'named_entities': 'sum',
    })
    
    return grouped_transcript_df


def add_entity_type_columns(transcript_df, entity_types):
    """
    This function creates new columns in a dataframe for each specified entity type.
    
    Parameters:
    transcript_df (pd.DataFrame): The dataframe to modify.
    entity_types (List[str]): The entity types to create columns for.
    
    Returns:
    transcript_df (pd.DataFrame): The modified dataframe with new entity type columns.
    """
    # Add columns for each entity type
    for ent_type in entity_types:
        transcript_df[ent_type.lower()] = transcript_df["named_entities"].apply(lambda x: \
                                                                                [entity.split('/')[0] for entity in x if entity.split('/')[1] == ent_type])
    return transcript_df

