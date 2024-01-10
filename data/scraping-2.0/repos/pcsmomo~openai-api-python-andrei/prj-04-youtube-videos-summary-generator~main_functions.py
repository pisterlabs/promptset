import sys
from pathlib import Path

# in jupyter (lab / notebook), based on notebook path

# print(f"Path.cwd(): {Path.cwd()}")
module_path = str(Path.cwd().parents[0])

if module_path not in sys.path:
    sys.path.append(module_path)

from common.usage import print_token_usage


# downloading the YouTube audio stream
def youtube_audio_downloader(link):    
    from pytube import YouTube
    import os
    import re
    if 'youtube.com' not in link:
        print('Invalid YouTube link!')
        return False
    
    yt = YouTube(link)
    
    audio = yt.streams.filter(only_audio=True).first()
    print('Downloading the audio stream ...', end='')
    output_file = audio.download()
    if os.path.exists(output_file):
        print('Done!')
    else:
        print('Error downloading the file!')
        return False
    
    basename = os.path.basename(output_file)
    name, extension = os.path.splitext(basename)
    audio_file = f'{name}.mp3'
    audio_file = re.sub('\s+', '-', audio_file)
    os.rename(basename, audio_file)
    return audio_file


# transcribing the audio_file or translating it to English
def transcribe(audio_file, not_english=False):
    import os
    import openai
    
    if not os.path.exists(audio_file):
        print('Audio file does not exist!')
        return False
    
    if not_english:  
        # translation to english
        with open(audio_file, 'rb') as f:
            print('Starting translating to English ...', end='')
            transcript = openai.Audio.translate('whisper-1', f)
            print('Done!')
    else:
        # transcription
        with open(audio_file, 'rb') as f:
            print('Starting transcribing ... ', end='')
            transcript = openai.Audio.transcribe('whisper-1', f)
            print('Done!')
        
    name, extension = os.path.splitext(audio_file)
    transcript_filename = f'transcript-{name}.txt'
    with open(transcript_filename, 'w') as f:
        f.write(transcript['text'])
            
    return transcript_filename


# summarizing the transcript using GPT
def summarize(transcript_filename):
    import openai 
    import os
    
    if not os.path.exists(transcript_filename):
        print('The transcript file does not exist!')
        return False
    
    with open(transcript_filename) as f:
        transcript = f.read()
        
    system_prompt = 'I want you to act as a Life Coach.'
    prompt = f'''Create a summary of the following text.
    Text: {transcript}
    
    Add a title to the summary.
    Your summary should be informative and factual, covering the most important aspects of the topic.
    Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
    by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.'''
    
    print('Starting summarizing ... ', end='')
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=2048,
        temperature=1
    )
    
    print('Done')
    print_token_usage(response)
    r = response['choices'][0]['message']['content']
    return r