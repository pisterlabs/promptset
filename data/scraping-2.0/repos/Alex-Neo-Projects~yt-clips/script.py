import os
import sys
import subprocess
import whisper
import openai

def download_youtube(link):
    arg = f"bash ./scripts/download_youtube.sh {link}"
    proc = subprocess.Popen([arg], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    return out.decode("utf-8").split("\n")[0]

def download_whisper(uuid):
    arg = f"bash ./scripts/download_whisper.sh {uuid}"
    proc = subprocess.Popen([arg], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

def make_summarization(path, uuid):
    text_output = read_text_file(path)

    output_summarize_path = f"./summarize/{uuid}.txt"
    # Uncomment line under to store summarization to a file
    # sys.stdout = open(output_summarize_path, 'a+')

    openai.api_key=""
    for res in openai.Completion.create(
        model="text-davinci-003",
        prompt=f"I will provide you with transcript from a video. Gime a me a one sentence TLDR of the transcript. Then extract the most important key points and use them as markdown formatted headings. Give a detailed extractive and abstract summary for each key point.  It is important that you are very specific and clear in your response. Conclude with a one paragraph abstract summary of what the video wanted to convince us of. \n\nVideo transcript:\n{text_output}\nSummary:",
        max_tokens=1000,
        temperature=0,
        stream=True
    ):
        sys.stdout.write(res.choices[0].text)
        sys.stdout.flush()
    print('\n')
    
def read_text_file(path):
    f = open(path, "r")
    return f.read()

def create_text_file(path):
    f = open(path, "a+")
    return f

# def find_num_tokens(path): 
#     output = read_text_file(path)
#     return len(output)//4

# def break_into_chunks(path): 
#     num_tokens = find_num_tokens(path)
#     text_file_output = read_text_file(path)
#     output = [text_file_output[i:i+2000] for i in range(0, 5)]
#     return output

def main():
    # download youtube audio using yt-dlp
    print('Downloading Youtube Video')
    youtube_audio_uuid = download_youtube('https://www.youtube.com/clip/UgkxmZ_575WLr_y6dkXJ60F9U2a310aB63D6')
    
    # convert to transcript using whisper
    print('Converting Video to Transcript')
    whisper_audio_path = download_whisper(youtube_audio_uuid)

    # use chat-gpt to summarize the whisper output
    print('Summarizing transcript:')
    make_summarization(f'./whisper-downloads/{youtube_audio_uuid}/{youtube_audio_uuid}.mp3.txt', youtube_audio_uuid)

if __name__ == "__main__":
    main()