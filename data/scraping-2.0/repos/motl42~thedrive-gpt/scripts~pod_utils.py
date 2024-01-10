
import soundfile as sf
import librosa
import whisper
from pyannote.audio import Pipeline
import wave
import contextlib
import ffmpeg
import concurrent
import os

from os import path
import json
import time

import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)


def create_log_id(log_name):
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_id = f"{log_name}-{current_datetime}"
    return log_id


def get_logger(**kwargs):

    if 'name' in kwargs:
        name = kwargs['name']
        log_id = create_log_id(name)

    if 'id' in kwargs:
        log_id = kwargs['id']
        if log_id is not None:
            # get process id
            pid = os.getpid()
            if log_id in logging.root.manager.loggerDict:
                return logging.getLogger(log_id), log_id

    # Create a logs folder if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_name = log_id.split("-")[0]

    # Set up the logger
    logger = logging.getLogger(log_id)
    logger.setLevel(logging.INFO)

    # Create a file handler for logging to a file
    log_file = f'logs/{log_id}.log'
    file_handler = logging.FileHandler(log_file)

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        f'%(asctime)s - p%(process)d - {log_name} - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger, log_id

from faster_whisper import WhisperModel
import namedtupled



def get_whiser_cpp_model(num_workers):

    model_path = "whisper-medium-ct2/"

    # Run on GPU with FP16

    whisper_model_cpp = WhisperModel(model_path, device="cuda", compute_type="float16", num_workers=num_workers)

    whisper_model_cpp._transcribe = whisper_model_cpp.transcribe

    def whiser_cpp_transcribe(file_path, **kwargs):
        _segments, audio_info = whisper_model_cpp._transcribe(file_path, language="en", beam_size=5, word_timestamps=True)
        segments = []
        for segment in _segments:
            segments.append(segment)
        result = {
            "segments": namedtupled.reduce(segments),
        }
        return result

    whisper_model_cpp.transcribe = whiser_cpp_transcribe
    return whisper_model_cpp
    
def get_podcasts(dir = ""):
    with open(os.path.join(dir, "data/thedrive.json"), "r") as f:
        podcasts = json.load(f)
    
    podcasts = [podcast for podcast in podcasts if not( "AMA" in podcast["title"] or ("Qualy" in podcast["title"]) or ("rebroadcast" in podcast["title"].lower()))]
    return podcasts


import json

podcasts = get_podcasts()

import openai
import json

def get_podcast_guests(titles):
    prompt ='''
                You are one of the best NLP models specialised on unstructed data parsing. Can you figure out the name of the guests from the podcast "thedrive" from Peter Attia based on the titles. 
                If you there is no guest, please answer "none".
                Please answer in this format: "<guest>".

                Example: 
                TITLEs: ['#244 ‒ The history of the cell, cell therapy, gene therapy, and more | Siddhartha Mukherjee', '#14 - Robert Lustig, M.D., M.S.L.: fructose, processed food, NAFLD, and changing the food system', 'Qualy #125 - Hierarchies in healthcare, physician burnout, and a broken system', 'The one-year anniversary episode with Olivia Attia: Reflecting on the past year and looking forward to exciting times ahead']
                GUESTS: ['Siddhartha Mukherjee', 'Robert Lustig', 'None', 'Olivia Attia']

                TITLES: {titles}
                '''.format(titles=titles).replace("'", '"')

    res = openai.Completion.create(
              max_tokens=960,
              temperature=0.5,
              model="text-davinci-003",
              prompt=prompt,
          )
    
    guests = json.loads(res['choices'][0]['text'].strip().split("GUESTS: ")[1])
    
    return guests


class PodHandler:

    def __init__(self, **kwargs):
        self.logger = kwargs.get("logger", logging)
        self.whisper_model = kwargs.get(
            "whisper_model", get_whiser_cpp_model(4))

    def seconds_to_str(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def get_audio_length(self, audio_file):
        with contextlib.closing(wave.open(audio_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            length = frames / float(rate)
            return length

    def mp3_to_wav(self, mp3_file):
        # Replace the ".mp3" ending with ".wav"
        wav_file = mp3_file.replace(".mp3", ".wav")

        # Load the audio data using librosa
        """ audio_data, sample_rate = librosa.load(mp3_file)
        f = open(wav_file, "wb")

        # Write the audio data to a WAV file using soundfile
        sf.write(wav_file, audio_data, sample_rate) """

        (ffmpeg.input(mp3_file, threads=0)
            .output(wav_file, ar=16000)
            .overwrite_output()
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True))

        # Return the complete file path for the new WAV file
        return wav_file

    def transcribe_audio_file(self, file_path, **kwargs):
        output_filename = path.join(path.dirname(
            file_path), "transcript_without_speaker.json")
        if (not kwargs.get("force", False)) and os.path.isfile(output_filename):
            with open(output_filename, 'r') as f:
                return json.load(f)
        length = self.get_audio_length(file_path)
        start_time = time.time()
        self.logger.info("Starting transcribing audio file {}".format(file_path))
        result = self.whisper_model.transcribe(
            file_path, language="en", word_timestamps=True)
        end_time = time.time()
        duration = time.time() - start_time
        speed = length / duration

        self.logger.info("Finished transcription in  {:.2f} seconds with a speed of {:.0f} of {} length".format(
             duration, speed, file_path))
        # save result to json file in the same folder as the audio file
        with open(output_filename, 'w') as f:
            json.dump(result, f, indent=4)

        return result

    # log start end end and the time it took

    def diarize_audio_file(self, file_path):
        output_filename = path.join(
            path.dirname(file_path), "diarization.json")
        if os.path.isfile(output_filename):
            with open(output_filename, 'r') as f:
                return json.load(f)
        start_time = time.time()
        self.logger.info("Starting diarizing audio file")
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization', use_auth_token="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")
        diarization = pipeline(file_path)
        diarization._tracks
        # diaryzation._tracks example SortedDict({<Segment(0.497812, 78.9328)>: {'A': 'SPEAKER_00'}, <Segment(79.6416, 107.266)>: {'B': 'SPEAKER_00'}})
        # convert diarization.__tracks to a list of dicts with start and end times and speaker
        segments = []
        counter = 0
        for segment, speaker in diarization._tracks.items():
            start = segment.start
            if counter == 0:
                start = 0
            end = segment.end
            speaker = list(speaker.values())[0]
            new_segment = {"start": start, "end": end, "speaker": speaker}
            segments.append(new_segment)
            counter += 1
        end_time = time.time()
        self.logger.info("Finished diarizing audio file in {:.2f} seconds".format(
            end_time - start_time))

        with open(output_filename, 'w') as f:
            json.dump(segments, f, indent=4)
        return segments

    def match_transcript_with_diar(self, transcript, diarization):

        # create a new json file based on diarization.txt and result.json
        output = []
        current_speaker = None
        current_text = ""
        current_words = []
        current_start = None
        current_end = None

        # get the first two elements of a dict

        for segment in transcript["segments"]:
            for word in segment["words"]:

                word_start = word["start"]
                word_end = word["end"]
                word_text = word["word"]
                word_speaker = None
                distances = {}
                for index, line in enumerate(diarization):
                    start = line["start"]
                    end = line["end"]
                    speaker = line["speaker"]
                    if start <= word_start <= end:
                        distances[index] = word_start - \
                            start + end - word_start

                # get smallest distance
                if len(distances) > 0:
                    smallest_distance = min(distances, key=distances.get)
                    word_speaker = diarization[smallest_distance]["speaker"]
                if word_speaker is None:
                    continue

                # if current_speaker is not none

                if current_speaker == word_speaker:
                    current_text += " " + word_text
                    current_words.append((word_text , word_start))
                    current_end = word_end
                    continue

                if current_speaker is not None:
                    output.append({
                        "speaker": current_speaker,
                        "text": current_text,
                        "start": current_start,
                        "end": current_end,
                        "words": current_words
                    })

                current_speaker = word_speaker
                current_text = word_text
                current_words = [(word_text , word_start)]
                current_start = word_start
                current_end = word_end
        output.append({
            "speaker": current_speaker,
            "text": current_text,
            "start": current_start,
            "end": current_end,
            "words": current_words
        })

        return output

    def transcribe_audio_file_with_speaker(self, audio_file):
        try:

            file_name = "/".join(audio_file.split("/")[-2:])
            self.logger.info(
                "start transcribing audio file: {}".format(file_name))
            # only if file ends with mp3 convert to wav
            wav_file = self.mp3_to_wav(
                audio_file) if audio_file.endswith('.mp3') else audio_file
            # print how long get_audio_length takes

            start_time = time.time()
            """ length = self.get_audio_length(wav_file) """

            """ with concurrent.futures.ThreadPoolExecutor() as executor:
                transcript_without_speaker_exec = executor.submit(
                    self.transcribe_audio_file, wav_file)
                diarization_exec = executor.submit(
                    self.diarize_audio_file, wav_file)
                transcript_without_speaker = transcript_without_speaker_exec.result()
                diarization = diarization_exec.result() """

            transcript_without_speaker = self.transcribe_audio_file(wav_file)
            diarization = self.diarize_audio_file(wav_file)

            transcript_file = path.join(path.dirname(audio_file), "transcript.json")
            if os.path.isfile(transcript_file):
                with open(transcript_file, 'r') as f:
                    transcript = json.load(f)
            else:
                transcript = self.match_transcript_with_diar(
                    transcript_without_speaker, diarization)

            duration = time.time() - start_time
           

            self.logger.info("Finished transcription with speakers of {} length in {:.2f} seconds".format(
                file_name, duration))

            with open(transcript_file, 'w') as f:
                json.dump(transcript, f, indent=4)

            return transcript
        except Exception as e:
            self.logger.error(
                f"Error while transcription of {audio_file}:", str(e))
