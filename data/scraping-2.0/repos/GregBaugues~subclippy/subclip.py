import os
import assemblyai as aai
from datetime import timedelta
import json
from openai import OpenAI
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pytube import YouTube
import subprocess

YOUTUBE_URL = "https://www.youtube.com/watch?v=_GxrDjGRfFc"
BASE_FILENAME = "new_heights"


def video_filename():
    return f"source_videos/{BASE_FILENAME}.mp4"


def video_only_filename():
    return f"source_videos/{BASE_FILENAME}_video_only.mp4"


def audio_filename():
    return f"audio/{BASE_FILENAME}.mp3"


def data_filename():
    return f"data/{BASE_FILENAME}.json"


def rendered_filename():
    return f"rendered/{BASE_FILENAME}.mp4"


def write_data(data):
    with open(data_filename(), "w") as f:
        json.dump(data, f, indent=4)


def load_data():
    with open(data_filename(), "r") as f:
        return json.load(f)


def clip_filename(i):
    return f"clips/{BASE_FILENAME}_{str(i).zfill(3)}.mp4"


def merge_audio_and_video():
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        video_only_filename(),
        "-i",
        audio_filename(),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        video_filename(),
    ]
    subprocess.run(ffmpeg_command, check=True)


def download_1080p(url=YOUTUBE_URL):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension="mp4", res="1080p").first()
    video.download(filename=video_only_filename())
    merge_audio_and_video()


def download_720p(url=YOUTUBE_URL):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension="mp4", res="720p").first()
    video.download(filename=video_filename())


def download_video(res="720p"):
    download_720p()
    extract_audio()
    if res == "1080p":
        download_1080p()


def extract_audio(infile=video_filename(), outfile=audio_filename()):
    command = f"ffmpeg -i {infile} -vn -acodec libmp3lame {outfile}"
    subprocess.run(command, shell=True)


def to_timestamp(ms):
    td = timedelta(milliseconds=ms)
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(
        hours, minutes, seconds, td.microseconds // 1000
    )


def transcribe():
    aai.settings.api_key = os.environ.get("AAI_API_KEY")
    config = aai.TranscriptionConfig(speaker_labels=True, auto_highlights=True)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_filename())
    print(transcript)
    return transcript


def clean_string(s):
    s = s.lower()
    s = "".join(c for c in s if c.isalnum() or c.isspace() or c == "'")
    return s


def get_transcript_data(transcript):
    data = {}
    data["youtube_url"] = YOUTUBE_URL
    data["transcript_id"] = transcript.id
    data["transcript"] = transcript.text
    data["duration"] = transcript.audio_duration

    data["utterances"] = []

    for utterance in transcript.utterances:
        data["utterances"].append(
            {
                "start": utterance.start,
                "end": utterance.end,
                "speaker": utterance.speaker,
                "duration": int(utterance.end) - int(utterance.start),
                "text": utterance.text,
            }
        )

    data["words"] = []
    for word in transcript.words:
        data["words"].append(
            {
                "text": clean_string(word.text),
                "start": word.start,
                "end": word.end,
                "confidence": word.confidence,
            }
        )

    data["highlights"] = []
    for result in transcript.auto_highlights.results:
        timestamps = []
        for t in result.timestamps:
            timestamps.append({"start": t.start, "end": t.end})

        data["highlights"].append(
            {
                "text": result.text,
                "count": result.count,
                "rank": result.rank,
                "timestamps": timestamps,
            }
        )

    return data


def ask_gpt(transcript, prompt=""):
    MODEL = "gpt-4-1106-preview"
    client = OpenAI()

    sys_msg = f"""
{prompt}
I'll tip you $2000 if the clip you return goes viral. 
(But you'll get no tip if you modify the quote -- it has to be an exact quote)
"""
    sys_msg += """
Return results in JSON in this format: 
{"phrases": ["What is your name?"]}
"""

    messages = [
        {"role": "system", "content": sys_msg},
    ]

    messages.append({"role": "user", "content": transcript})
    print("Asking GPT...", messages)
    response = client.chat.completions.create(
        model=MODEL, response_format={"type": "json_object"}, messages=messages
    )
    str_response = response.choices[0].message.content
    data = json.loads(str_response)
    return data


def get_phrases(data, prompt=None):
    if not data.get("phrases"):
        data["phrases"] = []
    new_phrases = ask_gpt(data["transcript"], prompt=prompt)
    for p in new_phrases["phrases"]:
        data["phrases"].append({"text": p})

    write_data(data)
    return data


def calc_durations(data):
    for i in range(len(data["utterances"])):
        p = data["utterances"][i]
        p["duration"] = int(p["end"]) - int(p["start"])
        data["utterances"][i] = p

    for i in range(len(data["words"])):
        w = data["words"][i]
        w["duration"] = int(w["end"]) - int(w["start"])
        data["words"][i] = w

    return data


def find_exact_stamp(data, phrase):
    # Clean up the phrase text.
    phrase_text = clean_string(phrase["text"])
    phrase_words = phrase_text.split()

    # Early exit if phrase is empty.
    if not phrase_words:
        return None, None

    # Iterate through words in data to find the matching phrase.
    for i in range(len(data["words"]) - len(phrase_words) + 1):
        if all(
            data["words"][i + j]["text"] == phrase_words[j]
            for j in range(len(phrase_words))
        ):
            phrase_start = int(data["words"][i]["start"])
            phrase_end = int(data["words"][i + len(phrase_words) - 1]["end"])

            if phrase_end < phrase_start:
                raise Exception(
                    f"ERROR: End time {phrase_end} is less than start time {phrase_start} for phrase:\n{phrase_text}"
                )

            return phrase_start, phrase_end

    # Phrase not found.
    print(f"ERROR: Could not find exact stamp for phrase:\n{phrase_text}")
    return None, None


def calc_word_frequency(data):
    words = data["words"]
    word_frequency = {}
    for word in words:
        w = clean_string(word["text"])
        if w in word_frequency:
            word_frequency[w] += 1
        else:
            word_frequency[w] = 1

    for w in data["words"]:
        w["frequency"] = word_frequency[clean_string(w["text"])]

    # print word frequency sorted by frequency

    # for w in sorted(word_frequency, key=word_frequency.get, reverse=True):
    #     if len(w) > 4 and word_frequency[w] > 5:
    #         print(w, word_frequency[w])

    return data


def stitch_clips():
    import os
    import subprocess

    clips_dir = "clips/"
    clips = [
        clips_dir + clip
        for clip in os.listdir(clips_dir)
        if clip.endswith(".mp4") and clip.startswith(BASE_FILENAME)
    ]
    clips.sort()

    with open("file_list.txt", "w") as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "concat",
            "-i",
            "file_list.txt",
            "-c",
            "copy",
            rendered_filename(),
        ]
    )
    os.remove("file_list.txt")


def slice_video(source, start, end, buffer=50, filename=video_filename()):
    if not filename:
        raise Exception("Filename is required")

    start = (start - buffer) / 1000
    end = (end + buffer) / 1000
    if start < 0:
        start = 0
    print("Slicing video from", start, " to ", end, "into", filename)
    command = [
        "ffmpeg",
        "-i",
        source,
        "-ss",
        str(start),
        "-to",
        str(end),
        "-reset_timestamps",
        "1",
        filename,
    ]
    subprocess.run(command, check=True)


def slice_by_words(words, buffer=50):
    for i, w in enumerate(words):
        slice_video(
            video_filename(),
            w["start"],
            w["end"],
            buffer=buffer,
            filename=clip_filename(i),
        )


def slice_by_phrases(phrases, buffer=50):
    print(phrases)
    for i, p in enumerate(phrases):
        print(p)
        slice_video(
            video_filename(),
            p["start"],
            p["end"],
            buffer=buffer,
            filename=clip_filename(i),
        )


def slice_by_timestamps(timestamps=[], buffer=50):
    for i, t in enumerate(timestamps):
        slice_video(
            video_filename(),
            t["start"],
            t["end"],
            buffer=buffer,
            filename=clip_filename(i),
        )


def find_words(data, needles):
    needles = [needles].flatten()
    found = []
    for w in data["words"]:
        if w["text"].lower() in needles:
            found.append(w)
    return found


def get_words_to_make_phrase(data, phrase):
    word_list = []
    phrase = phrase.lower()
    for w in phrase.split(" "):
        words = find_words(data, w)
        if not words:
            raise Exception("Could not find word: ", w)
        # iterate over words and add the one with highest confidence to the word_list
        max_duration = 0
        for word in words:
            if word["duration"] > max_duration:
                max_duration = word["duration"]
                best_word = word
        word_list.append(best_word)
    return word_list


def get_timestamps_for_highlights(data):
    timestamps = []
    for h in data["highlights"]:
        for t in h["timestamps"]:
            timestamps.append(
                {
                    "start": t.get("start"),
                    "end": t.get("end"),
                }
            )
    return timestamps


def get_timestamps_for_phrases(data):
    for i, p in enumerate(data["phrases"]):
        start, end = find_exact_stamp(data, p)
        if start and end:
            p["start"] = int(start)
            p["end"] = int(end)
            data["phrases"][i] = p
        else:
            print("Could not find exact stamp for phrase: ", p["text"])
            del data["phrases"][i]
    return data


def reset_phrases(data):
    try:
        del data["phrases"]
    except:
        pass
    return data


def clip_and_stitch_from_needles(data, needles=""):
    word_list = []
    for needle in needles.split(" "):
        words = find_words(data, needle)
        word_list.extend(words)

    # sort word_list by word['start']
    word_list.sort(key=lambda x: int(x["start"]))
    slice_by_words(word_list, buffer=100)
    stitch_clips()


def clip_and_stitch_to_make_phrase(data, phrase):
    words = get_words_to_make_phrase(data, phrase)
    slice_by_words(words, buffer=50)


def clip_and_stitch_from_highlights(data):
    timestamps = get_timestamps_for_highlights(data)
    slice_by_timestamps(timestamps, buffer=50)
    stitch_clips()


def clip_and_stitch_from_prompt(data, prompt=None):
    # data = reset_phrases(data)
    if not data.get("phrases"):
        data["phrases"] = []
        data = get_phrases(data, prompt=prompt)
        write_data(data)

    data = get_timestamps_for_phrases(data)
    write_data(data)

    slice_by_phrases(data["phrases"], buffer=150)
    stitch_clips()


if __name__ == "__main__":
    if not os.path.exists(video_filename()):
        download_video(res="1080p")

    if os.path.exists(data_filename()):
        data = load_data()
    else:
        transcript = transcribe()
        data = get_transcript_data(transcript)
        write_data(data)

    prompt = """
            This is a transcript from a youtube video.
            Extract the most interesting and funny quotes from this clip. 
            Give me exact quotes -- do not paraphrase.
            Select the clips most likely to go viral.
            Each clip should be 50-200 words.
            """

    clip_and_stitch_from_prompt(data, prompt=prompt)
    # clip_and_stitch_from_needles(data, needles=["lazers"])
    # clip_and_stitch_from_phrase(data, phrase="")
    # clip_and_stitch_from_highlights(data)
