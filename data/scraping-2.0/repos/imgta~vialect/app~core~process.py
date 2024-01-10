from core.utils import TaskUtility
from datetime import datetime
from config import MEDIA_LIB
from yt_dlp import YoutubeDL
from typing import Optional
from pathlib import Path
import streamlit as st
import subprocess
import mimetypes
import time
import json
import os


class AudioProcess:
    tU = TaskUtility()

    @staticmethod
    def mono_resample(input_file_path: str) -> Optional[str]:
        """[RESAMPLING] -> Reduce audio to mono-channel, resample to 16kHz audio for downstream diarization. (.OGG or .mp3)"""
        base_name, _ = os.path.splitext(input_file_path)
        output_path = f"{base_name}.OGG"
        ffmpeg_cmds = ["ffmpeg", "-i", input_file_path, "-vn", "-sn", "-dn", "-ar", "16000", "-ac", "1", output_path]
        try:
            subprocess.run(ffmpeg_cmds, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Converted {input_file_path} to mono channel, resampled at 16kHz.")
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion: {e}")
            return
        except subprocess.TimeoutExpired:
            print(f"FFmpeg command timed out for file: {input_file_path}")
            return
        return output_path

    def extract_audio(self, video_url: str) -> tuple[str, float]:
        """[AUDIO FROM VIDEO URL] -> Extract, convert audio from video"""
        start_time = time.time()
        try:
            with YoutubeDL({'quiet': True, 'noplaylist': True}) as ydl:
                yt_dict = ydl.extract_info(url=video_url, download=False)

            # Extract and save relevant video info in json
            key_info = ['id', 'title', 'extractor_key', 'webpage_url', 'language', 'thumbnail', 'tags', 'categories', 'description', 'uploader', 'uploader_url','duration_string', 'upload_date', 'view_count']
            info_dict = {key: yt_dict.get(key, None) for key in key_info}

            extract_date = datetime.now().strftime("%b %d, %Y %I:%M %p").replace(" 0", " ")
            info_dict['extract_date'] = extract_date

            for key in key_info:
                if key in yt_dict:
                    info_dict[key] = yt_dict[key]

            # Create media directory from video name
            video_title = self.tU.sanitize_name(info_dict.get('title', 'audio'))
            MEDIA_PATH = Path(MEDIA_LIB) / video_title
            os.makedirs(MEDIA_PATH, exist_ok=True)
            info_dict['title_dir'] = video_title

            info_file = MEDIA_PATH / "info.json"
            with open(file=info_file, mode='w', encoding='utf-8') as inf:
                json.dump(info_dict, inf, ensure_ascii=True)

            full_info = MEDIA_PATH / "full_info.json"
            with open(file=full_info, mode='w', encoding='utf-8') as full:
                json.dump(yt_dict, full, ensure_ascii=True)

            # Check if audio file already exists in the folder
            audio_files = list(MEDIA_PATH.glob('audio*.OGG'))
            if audio_files:
                AUDIO_FILE = str(audio_files[0])
                st.toast(body="Existing audio file found!", icon="✔")
                return AUDIO_FILE, time.time() - start_time
            else:
                # Extract and download audio, convert to mono and resample to 16kHz for Pyannote ingestion
                ytdl_options = {
                    'format': 'worstaudio/worst',
                    'outtmpl': os.path.join(MEDIA_PATH, 'audio.%(ext)s'),
                }
                try:
                    with YoutubeDL(ytdl_options) as ytdl:
                        result = ytdl.extract_info(url=video_url, download=True)
                        AUDIO_PATH = ytdl.prepare_filename(result)
                    processed_audio = self.mono_resample(AUDIO_PATH)
                    return processed_audio, time.time() - start_time
                except Exception as e:
                    raise Exception(f"Error extracting audio: {e}")
        except Exception as e:
            raise Exception(f"Error extracting audio: {e}")


    def process_upload(self, file_path: str) -> tuple[str, float]:
        """[AUDIO FROM UPLOAD] -> Identify, then process upload file for audio extraction"""
        start_time = time.time()
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith("audio"):
            processed_audio = self.mono_resample(file_path)
            return processed_audio, time.time() - start_time
        elif mime_type and mime_type.startswith("video"):
            return self.extract_audio(file_path)
        else:
            raise ValueError("Unsupported file type")


from pyannote.audio import Pipeline
from langcodes import Language
import whisper
import openai
import torch


class AudioTransform:
    tU = TaskUtility()
    devices = torch.device("cuda" if tU.has_cuda() else "cpu")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=st.session_state['hf_access_token'])

    def diarize_audio(self, audio_path) -> tuple[str, float]:
        """[SPEAKER DIARIZATION] => Partition audio stream to id speaker segments, generate rich transcription time marked (RTTM)
        Note: Use 'speaker-diarization@2.1' as a fallback. Version 3.1 can now utilize cuda devices (GPU)."""
        start_time = time.time()
        MEDIA_DIR = Path(audio_path).parent
        rttm_output = MEDIA_DIR / "speakers.rttm"
        # Check if RTTM file already exists
        if rttm_output.exists():
            st.toast(body="Existing RTTM found!", icon="✔")
            return rttm_output, time.time() - start_time
        else:
            try:
                # Establish pipeline device usage
                self.pipeline.to(self.devices)
                diarization = self.pipeline(audio_path, min_speakers=1, max_speakers=5)

                with open(rttm_output, "w") as rttm:
                    diarization.write_rttm(rttm)
                return rttm_output, time.time() - start_time
            except Exception as e:
                st.error(body=f"Error during diarization: {e}")
                raise Exception(f"Error during diarization: {e}")


    @staticmethod
    def parse_rttm(file_path: str) -> tuple[list, float]:
        """[RTTM PARSING] -> Isolate speaker_ids and timestamps"""
        start = time.time()
        with open(file_path, "r") as rttm:
            lines = rttm.readlines()

        rttm_log = []
        for line in lines:
            part = line.strip().split()
            speaker, time_start, duration = part[7], part[3], part[4]
            rttm_log.append((speaker, time_start, duration))
        return rttm_log, time.time() - start


    @staticmethod
    def get_full_language(lang_code: str) -> str:
        """"[LANGUAGE ID] -> Converts language code to full language name"""
        language = Language.get(lang_code)
        return language.display_name()


    def scribe_audio(self, file_path: str, model_name: str, translate: bool) -> tuple[dict, float]:
        """[WHISPER TRANSCRIBING] -> Use OpenAI's whisper model to transcribe (+/- translate) audio to readable text"""
        start_time = time.time()
        SCRIPT_PATH = os.path.join(os.path.dirname(file_path), 'transcript.json')

        # Load or initialize list of transcripts
        try:
            with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)
            if not isinstance(transcripts, list):
                transcripts = []
        except (FileNotFoundError, json.JSONDecodeError):
            transcripts = []

        # Check for existing transcripts
        for entry in transcripts:
            if entry['model'] == model_name:
                return entry['full_script'], time.time() - start_time

        # Whisper transcription setup parameters
        whisper_model = whisper.load_model(name=model_name, device=self.devices)
        decode_lang = 'translate' if translate else None
        script = whisper_model.transcribe(audio=file_path, word_timestamps=True, task=decode_lang)
        lang = script['language'] if translate else None

        new_script = {
            'model': model_name,
            'translated': lang,
            'full_script': script,
        }
        transcripts.append(new_script)

        # Write updated transcripts back to transcript.json file
        with open(SCRIPT_PATH, 'w', encoding='utf-8') as f:
            json.dump(transcripts, f, ensure_ascii=True)
        return script, time.time() - start_time


    def merge_segments(self, dialog, tolerance: float = 0.0):
        merged_dialog = []
        prev_log = None

        for log in dialog:
            # Only merge when speaker_ids from both logs match
            if prev_log and prev_log['speaker'] == log['speaker']:
                prev_end = self.tU.convert_to_seconds(prev_log['end'])
                current_start = self.tU.convert_to_seconds(log['start'])

                # Find matching ending and starting time segments from previous and current logs, respectively
                if abs(prev_end - current_start) <= tolerance:
                    prev_log['text'] += f" {log['text'].strip()}"
                    prev_log['end'] = log['end'] # Keep end time of current segment
                    continue

            if prev_log:
                merged_dialog.append(prev_log)
            prev_log = log.copy()

        if prev_log:
            merged_dialog.append(prev_log)

        return merged_dialog


    def align_script(self, file_path: str, model_name: str, transcript: dict, rttm_data: list, tolerance: float = 1.5) -> tuple[list, float]:
        """[TRANSCRIPT ALIGNMENT] -> Synchronize, then align timestamps with speaker_ids + text
        Note: Whisper's generated timestamps are often times inaccurate and differ from their Pyannote counterparts.
        This is just a quick and dirty method of generating more accurate timestamps through segment comparisons and overlap detection, tempered by tolerance level."""
        exec_start = time.time()
        STAMP_PATH = os.path.join(os.path.dirname(file_path), 'stamps.json')

        # Load existing or initialize new timestamps
        try:
            with open(STAMP_PATH, 'r', encoding='utf-8') as log:
                stamps = json.load(log)
        except (FileNotFoundError, json.JSONDecodeError):
            stamps = []

        # Check if timestamps exist for whisper model_name
        for entry in stamps:
            if entry['model'] == model_name:
                if 'timestamps' in entry:
                    return entry['timestamps'], time.time() - exec_start
                break

        dialog = []
        for segment in transcript['segments']:
            seg_start, seg_end = round(float(segment['start']), 4), round(float(segment['end']), 4)

            for speaker_id, start_time, duration in rttm_data:
                rttm_start, rttm_end = round(float(start_time), 4), round((float(start_time) + float(duration)), 4)

                # Overlap detection between Whisper and Pyannote segments
                if (rttm_start - tolerance <= seg_start <= rttm_end + tolerance) or (rttm_start - tolerance <= seg_end <= rttm_end + tolerance):
                    top_speaker = speaker_id
                    break

            if top_speaker is not None:
                hms_start = self.tU.format_timestamp(seg_start if abs(abs(seg_start - rttm_start) - tolerance) > 0.5 else rttm_start)
                hms_end = self.tU.format_timestamp(seg_end if abs(abs(seg_end - rttm_end) - tolerance) > 0.5 else rttm_end)

                dialog.append({
                    'start': hms_start,
                    'speaker': top_speaker,
                    'text': segment['text'].strip(),
                    'end': hms_end,
                })

        # Append and save new timestamp data in stamps.json
        merged_dialog = self.merge_segments(dialog, 0)
        stamps.append({'model': model_name, 'timestamps': merged_dialog})
        with open(STAMP_PATH, 'w', encoding='utf-8') as log:
            json.dump(stamps, log, ensure_ascii=True)

        return merged_dialog, time.time() - exec_start


    def summarize(self,transcript_text: str, file_path: str, model_name: str) -> str:
        """[SUMMARIZATION] -> Generate a summary in JSON format via a gpt model from full transcript text"""
        SCRIPT_PATH = os.path.join(os.path.dirname(file_path), 'transcript.json')
        GPT_MODEL = "gpt-3.5-turbo-1106"

        try:
            with open(SCRIPT_PATH, 'r', encoding='utf-8') as script:
                transcripts = json.load(script)
        except (FileNotFoundError, json.JSONDecodeError):
            transcripts = []

        # Check for existing summary generated for selected whisper model
        for entry in transcripts:
            if entry['model'] == model_name:
                if 'summary' in entry and GPT_MODEL in entry['summary']:
                    return entry['summary'][GPT_MODEL]

        # OpenAI ChatGPT client settings
        client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
        context = "You are skilled in summarizing key ideas/concepts from audio/video transcripts in JSON output: { 'summary': <summary here> }"
        response = client.chat.completions.create(
            model=GPT_MODEL,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": f"Summarize the following text:\n{transcript_text}"}
            ],
            temperature=0.2, # creativity, [0 -> 2]
            max_tokens=250, # limits response length
            # top_p=0.2, # vocabulary diversity, [0 -> 1]
            # frequency_penalty=0.5, # penalizes repition, [0 -> 2]
            # presence_penalty=0.5, # encourages new topics, [0 -> 2]
        )
        json_res = response.choices[0].message.content
        summary = json.loads(json_res)['summary']
        vial_date = datetime.now().strftime("%b %d, %Y %I:%M %p").replace(" 0", " ")

        # Append generated summary and date to transcript
        for entry in transcripts:
            if entry['model'] == model_name:
                entry['summary'] = {GPT_MODEL: summary}
                entry['vial_date'] = vial_date
                break

        with open(SCRIPT_PATH, 'w', encoding='utf-8') as f:
            json.dump(transcripts, f, ensure_ascii=True)

        return summary


    def text2speech(self, text_input: str, output_dir, whisper_model: str):
        OUTPUT_PATH = os.path.join(output_dir, f"{whisper_model}_tts.mp3")
        TTS_MODEL = 'tts-1'
        VOICE_MODEL = 'nova'

        client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=VOICE_MODEL,
            input=text_input,
        )
        response.stream_to_file(OUTPUT_PATH)
