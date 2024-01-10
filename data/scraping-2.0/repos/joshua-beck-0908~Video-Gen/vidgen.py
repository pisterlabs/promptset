import openai
from moviepy.editor import *
from bing_image_downloader import downloader
from elevenlabs import generate, play
import elevenlabs
import os
import sys
import argparse
import shutil
from pathlib import Path
import json

openai.api_key = os.environ['OPENAI_API_KEY']
elevenlabs.set_api_key(os.environ['ELEVENLABS_API_KEY'])
systemContext = """You are a helpful educator writing a script for a video. 
You divide your response into paragraphs and put the a title for each paragraph curly brackets at the start of the paragraph,
followed by simple related image search keywords (including prompt words) with the paragraph in square brackets, and finish the paragraph with three dashes.
Example: {Title} [image keywords] text of paragraph ---
"""

def main():
    parser = argparse.ArgumentParser(description='Generate a video from a text prompt.')
    parser.add_argument('prompt', help='The prompt to generate the video from.')
    
    args = parser.parse_args()
    prompt = args.prompt
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages = [
            {'role': 'system', 'content': systemContext},
            {'role': 'user', 'content': prompt}
        ],
    )
    paragraphs = response.choices[0]['message']['content'].split('---')
    tmpDir = Path('tmp')
    if Path('tmp').exists():
        shutil.rmtree(str(tmpDir), ignore_errors=True)
    tmpDir.mkdir()
    with open(tmpDir / 'prompt.txt', 'w') as f:
        f.write(prompt)
    with open(tmpDir / 'raw_response.txt', 'w') as f:
        json.dump(response.choices[0]['message'], f)
    with open(tmpDir / 'response.txt', 'w') as f:
        f.write('\n\n'.join(paragraphs))
    for n, par in enumerate(paragraphs):
        print(f'Paragraph {n+1}/{len(paragraphs)}')
        parDir = tmpDir / f'part{n:02}'
        parDir.mkdir()
        par = par.strip()
        if par == '':
            continue
        par = par.replace('{', '').split('}')
        title = par[0]
        with open(parDir / 'title.txt', 'w') as f:
            f.write(title)
        par = par[1].strip()
        print(f'Title: {title}')
                          
        par = par.replace('[', '', 1).split(']')
        imgKeywords = par[0]
        with open(parDir / 'keywords.txt', 'w') as f:
            f.write(imgKeywords)
        par = par[1].strip()
        print(f'Finding images for: {imgKeywords}')
        downloader.download(imgKeywords, limit=2, output_dir=parDir, verbose=False)

        with open(parDir / 'text.txt', 'w') as f:
            f.write(par)
            
        print(f'Generating audio...')
        audio = generate(text=par, voice='Charlie', model='eleven_monolingual_v1')
        elevenlabs.save(audio, parDir / 'audio.mp3')

    print('Done.')
    # https://stackoverflow.com/questions/70125988/how-do-i-create-a-video-from-a-static-image-and-an-audio-file


if __name__ == '__main__':
    main()

