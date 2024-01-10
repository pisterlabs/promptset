# from IPython.display import display, Markdown
import openai
import os
import subprocess
import yt_dlp
import whisper
import ffmpeg
import json
from .utils import write_srt, write_compact_srt, write_word_chunked_srts
openai.api_key = os.environ.get("OPENAI_API_KEY")

#####################
#     Variables     #

DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")  # or "cpu" or "mps" for mac
# or "medium.en" or "large.en"
MODEL = os.environ.get("WHISPER_MODEL", "small.en")
FFMPEG_HWACCEL = os.environ.get("FFMPEG_HWACCEL", "none")

#####################


def download_and_transcribe(url=None, workspace=None):
    workspace = workspace or "{workspace}"
    # Ensure that the necessary directories exist
    os.makedirs(workspace, exist_ok=True)

    # Ask for a YouTube URL
    # url = input("meow <3! enter your youtewb url: ")
    url = url or input("meow <3! enter your youtewb url: ")
    # url = "https://www.youtube.com/watch?v=xFWakbQAk5Q"

    print("Downloading video info...")
    with yt_dlp.YoutubeDL({
        'quiet': True,
    }) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info['id']

        # make workspace/<video_id>/(gen_temp, gen_final)
        os.makedirs(f"{workspace}/{video_id}/gen_temp", exist_ok=True)
        os.makedirs(f"{workspace}/{video_id}/gen_final", exist_ok=True)

    audio_path = f"{workspace}/{video_id}/gen_temp/audio.wav"
    video_path = f"{workspace}/{video_id}/gen_temp/video.mp4"
    ydl_opts = {
        'format': 'mp4/bestvideo+bestaudio',
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'outtmpl': video_path,
        'quiet': True,
    }

# check if video is already downloaded as mp4+audio
    if os.path.exists(video_path) and os.path.exists(audio_path):
        print("Video already downloaded as mp4+audio.")
    else:
        print("Downloading video as mp4...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([url])

        # extract mp3 from mp4

        print("Extracting audio from video...")
        # ffmpeg.input(video_path).output(audio_path, acodec="pcm_s16le", ac=1, ar="16k").run(quiet=True, overwrite_output=True)
        # run quietly but print errors, overwrite files
        output = subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec",
                                "pcm_s16le", "-ac", "1", "-ar", "16k", audio_path], capture_output=True)
        assert output.returncode == 0

    if os.path.exists(f"{workspace}/{video_id}/gen_final/subtitles.srt"):
        print("Subtitles already generated.")
        return video_id
    # https://www.youtube.com/watch?v=xFWakbQAk5Q
    print(f"Loading whisper model {MODEL} on device {DEVICE}...")
    model = whisper.load_model(MODEL, device=DEVICE)

    print("Transcribing audio...")
    result = model.transcribe(
        audio_path, language="en", word_timestamps=True, verbose=False)

    # print(result["segments"])
    print("Saving subtitles as txt/srt/csrt...")

    # save result["segments"] to srt file
    # save result["text"] to txt file

    with open(f"{workspace}/{video_id}/gen_final/subtitles.srt", "w", encoding="utf-8") as srt:
        write_srt(result["segments"], file=srt)

    with open(f"{workspace}/{video_id}/gen_final/subtitles.csrt", "w", encoding="utf-8") as csrt:
        write_compact_srt(result["segments"], file=csrt)

    with open(f"{workspace}/{video_id}/gen_final/subtitles.txt", "w", encoding="utf-8") as txt:
        txt.write(result["text"])

    write_word_chunked_srts(result["segments"], open(f"{workspace}/{video_id}/gen_final/subtitles_chunked.srt", "w", encoding="utf-8"), open(
        f"{workspace}/{video_id}/gen_final/subtitles_chunked.csrt", "w", encoding="utf-8"), chars_per_chunk=18)
    return video_id


def analyse_with_paste(video_id, workspace=None):
    workspace = workspace or "{workspace}"
    anal_json = f"{workspace}/{video_id}/gen_final/analysis.json"
    if os.path.exists(anal_json):
        print("Analysis already exists.")
        return
    json_input_lines = []

    print("Please enter JSON data line by line. Type 'done' on a new line to finish.")

    while True:
        line = input()
        if line.strip().lower() == 'done':
            break
        json_input_lines.append(line)

    json_input = '\n'.join(json_input_lines)

    try:
        data = json.loads(json_input)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}")
        exit(1)

    with open(anal_json, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def analyse_with_chatgpt(video_id, workspace=None):
    workspace = workspace or "{workspace}"
    system_prompt = """You are an analyser, which looks at the transcripts of podcasts and selects a few (max three) high quality 10-30second (must be AT LEAST 10 seconds, dont pick out like 15 short segments, only a few high quality ones) segments which would engage the most largest audience and most amount of viewers in a video clip on social media - audience would be age of between 13 and 24, and dont forget to consider things such as the main hooks in the transcript, and stuff like controversial opinions or interesting facts which could attract attention

You are to return a JSON array of objects, where in the object: "start" and "end" are the start/end points, "summary" is a one sentence (max 30word) summary of the clip section, and "reason" is a reason from you about why you picked this clip (max 15 words)

example:
[{
  "start":"0:00:28.300",
  "end": "0:00:45.620",
  "summary": "Freedom won't happen for the rest of the world, the divide will remain.",
  "reason": "Super controversial opinion - would engage audience/comments"
}]"""
    anal_json = f"{workspace}/{video_id}/gen_final/analysis.json"
    if os.path.exists(anal_json):
        print("Analysis already exists.")
        return

    file = open(f"{workspace}/{video_id}/gen_final/subtitles.csrt",
                "r", encoding="utf-8")
    transcript = file.read()
    print("Analyzing transcript with GPT3.5-Turbo...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": transcript}
                  ])

    usage = response["usage"]
    prompt = usage['prompt_tokens']
    comp = usage['completion_tokens']
    print(
        f"Used {prompt} prompt + {comp} completion ({usage['total_tokens']} total ~ ${(prompt/1000*0.0015) + (comp/1000*0.002)}) tokens.")
    items = response["choices"][0]["message"]["content"]

    # filter out segments that are too short
    items = [item for item in items if item["end"] - item["start"] > 5]
    for item in items:
        print(f"{item['start']} --> {item['end']}: {item['reason']}")

    with open(anal_json, "w", encoding="utf-8") as json:
        json.write(json.dumps(items))


def seperate_into_clips(video_id, workspace=None):
    workspace = workspace or "{workspace}"
    file = open(f"{workspace}/{video_id}/gen_final/analysis.json",
                "r", encoding="utf-8")
    analysis = file.read()
    print("Seperating video into clips...")
    os.makedirs(f"{workspace}/{video_id}/gen_final/clips", exist_ok=True)
    parsed = json.loads(analysis)
    for clip in parsed:
        start = clip["start"]
        end = clip["end"]
        summary = clip["summary"]
        print(f"Cutting clip from {start} to {end}...")
        # output = subprocess.run(["ffmpeg", "-y", "-i", f"{workspace}/{video_id}/gen_temp/video.mp4", "-ss", start, "-to", end, "-c", "copy", f"{workspace}/{video_id}/gen_final/clips/{summary}.mp4"], capture_output=True)
        # assert output.returncode == 0
        # print(f"Saved clip to {workspace}/{video_id}/gen_final/clips/{summary}.mp4")

        subtitle_path = f"{workspace}/{video_id}/gen_final/subtitles_chunked.srt"
        out_path = f"{workspace}/{video_id}/gen_final/clips/{summary}.mp4"
        # subtitles should be burned in, white text with black outline in the center of the screen, and crop to tiktok ratio (1080x1920) or smaller in the middle of the screen
        # output = subprocess.run(["ffmpeg", "-y", "-i", f"{workspace}/{video_id}/gen_temp/video.mp4", "-ss", start, "-to", end, "-vf", f"subtitles={subtitle_path}:force_style='OutlineColour=&H40000000,BorderStyle=3'", "-c:a", "copy", out_path], capture_output=True)
        # sub_style = "Alignment=10,OutlineColour=&H10000000,BorderStyle=3,Fontsize=12,MarginV=25"
        # sub_style = "Alignment=10,Fontname=Consolas,BackColor=&H00000000,PrimaryColour=&H00FFFFFF,Bold=1,Italid=1,OutlineColour=&H80000000,Spacing=0.2,Outline=0,Shadow=1,MarginV=250,BorderStyle=3,Fontsize=12"

        # center, transparent background, white text, black outline, crop to tiktok ratio, font size 12

        # sub_style = "Alignment=10,Fontname=Consolas,BackColor=&H00000000,PrimaryColour=&H00FFFFFF,Bold=1,Italic=1,Spacing=0.2,Outline=0,Shadow=0,BorderStyle=3,Fontsize=12"
        sub_style = "Alignment=10,Fontname=Trebuchet MS,BackColour=&H80000000,Spacing=0.2,Outline=0,Shadow=0.75,PrimaryColour=&H00FFFFFF,Bold=1,MarginV=250,Fontsize=16"
        output = subprocess.run(["ffmpeg",
                                 FFMPEG_HWACCEL and "-hwaccel" or "", FFMPEG_HWACCEL and FFMPEG_HWACCEL or "",
                                 "-y",
                                 "-i",
                                 f"{workspace}/{video_id}/gen_temp/video.mp4",
                                 "-ss", start, "-to", end,
                                 "-vf", f"subtitles={subtitle_path}:force_style='{sub_style}',crop=ih*(9/16):ih",
                                 "-c:a", "copy",  out_path], capture_output=True)
        if output.returncode != 0:
            print(output.stderr)
        assert output.returncode == 0

        # print(f"Saved clip with burned subs to {out_path}")


if __name__ == "__main__":
    # video_id = download_and_transcribe("https://www.youtube.com/watch?v=xFWakbQAk5Q")
    # video_id = download_and_transcribe()
    print("meow <3! you're using meow_ttai!")
    video_id = download_and_transcribe(
        'https://www.youtube.com/watch?v=RcYjXbSJBN8')
    # analyse_with_(video_id)
    seperate_into_clips(video_id)
