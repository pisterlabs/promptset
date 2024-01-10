# from faster_whisper import WhisperModel
# import datetime
# import subprocess
# from pathlib import Path
# import pandas as pd
# import re
# import time
# import os 
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import silhouette_score

# from pytube import YouTube
# import yt_dlp
# import torch
# import pyannote.audio
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
# from pyannote.audio import Audio
# from pyannote.core import Segment

# from gpuinfo import GPUInfo

# import wave
# import contextlib
# from transformers import pipeline
# import psutil

# from view import *

# whisper_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
# source_languages = {
#     "en": "English",
#     "zh": "Chinese",
#     "de": "German",
#     "es": "Spanish",
#     "ru": "Russian",
#     "ko": "Korean",
#     "fr": "French",
#     "ja": "Japanese",
#     "pt": "Portuguese",
#     "tr": "Turkish",
#     "pl": "Polish",
#     "ca": "Catalan",
#     "nl": "Dutch",
#     "ar": "Arabic",
#     "sv": "Swedish",
#     "it": "Italian",
#     "id": "Indonesian",
#     "hi": "Hindi",
#     "fi": "Finnish",
#     "vi": "Vietnamese",
#     "he": "Hebrew",
#     "uk": "Ukrainian",
#     "el": "Greek",
#     "ms": "Malay",
#     "cs": "Czech",
#     "ro": "Romanian",
#     "da": "Danish",
#     "hu": "Hungarian",
#     "ta": "Tamil",
#     "no": "Norwegian",
#     "th": "Thai",
#     "ur": "Urdu",
#     "hr": "Croatian",
#     "bg": "Bulgarian",
#     "lt": "Lithuanian",
#     "la": "Latin",
#     "mi": "Maori",
#     "ml": "Malayalam",
#     "cy": "Welsh",
#     "sk": "Slovak",
#     "te": "Telugu",
#     "fa": "Persian",
#     "lv": "Latvian",
#     "bn": "Bengali",
#     "sr": "Serbian",
#     "az": "Azerbaijani",
#     "sl": "Slovenian",
#     "kn": "Kannada",
#     "et": "Estonian",
#     "mk": "Macedonian",
#     "br": "Breton",
#     "eu": "Basque",
#     "is": "Icelandic",
#     "hy": "Armenian",
#     "ne": "Nepali",
#     "mn": "Mongolian",
#     "bs": "Bosnian",
#     "kk": "Kazakh",
#     "sq": "Albanian",
#     "sw": "Swahili",
#     "gl": "Galician",
#     "mr": "Marathi",
#     "pa": "Punjabi",
#     "si": "Sinhala",
#     "km": "Khmer",
#     "sn": "Shona",
#     "yo": "Yoruba",
#     "so": "Somali",
#     "af": "Afrikaans",
#     "oc": "Occitan",
#     "ka": "Georgian",
#     "be": "Belarusian",
#     "tg": "Tajik",
#     "sd": "Sindhi",
#     "gu": "Gujarati",
#     "am": "Amharic",
#     "yi": "Yiddish",
#     "lo": "Lao",
#     "uz": "Uzbek",
#     "fo": "Faroese",
#     "ht": "Haitian creole",
#     "ps": "Pashto",
#     "tk": "Turkmen",
#     "nn": "Nynorsk",
#     "mt": "Maltese",
#     "sa": "Sanskrit",
#     "lb": "Luxembourgish",
#     "my": "Myanmar",
#     "bo": "Tibetan",
#     "tl": "Tagalog",
#     "mg": "Malagasy",
#     "as": "Assamese",
#     "tt": "Tatar",
#     "haw": "Hawaiian",
#     "ln": "Lingala",
#     "ha": "Hausa",
#     "ba": "Bashkir",
#     "jw": "Javanese",
#     "su": "Sundanese",
# }

# source_language_list = [key[0] for key in source_languages.items()]

# def convert_time(secs):
#     return datetime.timedelta(seconds=round(secs))

# embedding_model = PretrainedSpeakerEmbedding( 
#     "speechbrain/spkrec-ecapa-voxceleb",
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# def speech_to_text(video_file_path, selected_source_lang, whisper_model, num_speakers):
#     """
#     # Transcribe youtube link using OpenAI Whisper
#     1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
#     2. Generating speaker embeddings for each segments.
#     3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    
#     Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
#     Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
#     """
    
#     # model = whisper.load_model(whisper_model)
#     model = WhisperModel(whisper_model) 
#     # model = WhisperModel(whisper_model, device="cpu", compute_type="int8_float16")

#     time_start = time.time()
#     if(video_file_path == None):
#         raise ValueError("Error no video input")
#     print(video_file_path)

#     try:
#         # Read and convert youtube video
#         _,file_ending = os.path.splitext(f'{video_file_path}')
#         print(f'file enging is {file_ending}')
#         audio_file = video_file_path.replace(file_ending, ".wav")
#         print("starting conversion to wav")
#         os.system(f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')
        
#         # Get duration
#         with contextlib.closing(wave.open(audio_file,'r')) as f:
#             frames = f.getnframes()
#             rate = f.getframerate()
#             duration = frames / float(rate)
#         print(f"conversion to wav ready, duration of audio file: {duration}")

#         # Transcribe audio
#         options = dict(language=selected_source_lang, beam_size=5, best_of=5)
#         transcribe_options = dict(task="transcribe", **options)
#         segments_raw, info = model.transcribe(audio_file, **transcribe_options)

#         # Convert back to original openai format
#         segments = []
#         i = 0
#         for segment_chunk in segments_raw:
#             chunk = {}
#             chunk["start"] = segment_chunk.start
#             chunk["end"] = segment_chunk.end
#             chunk["text"] = segment_chunk.text
#             segments.append(chunk)
#             i += 1
#         print("transcribe audio done with fast whisper")
#     except Exception as e:
#         raise RuntimeError("Error converting video to audio")

#     try:
#         # Create embedding
#         def segment_embedding(segment):
#             audio = Audio()
#             start = segment["start"]
#             # Whisper overshoots the end timestamp in the last segment
#             end = min(duration, segment["end"])
#             clip = Segment(start, end)
#             waveform, sample_rate = audio.crop(audio_file, clip)
#             return embedding_model(waveform[None])

#         embeddings = np.zeros(shape=(len(segments), 192))
#         for i, segment in enumerate(segments):
#             embeddings[i] = segment_embedding(segment)
#         embeddings = np.nan_to_num(embeddings)
#         print(f'Embedding shape: {embeddings.shape}')

#         if num_speakers == 0:
#         # Find the best number of speakers
#             score_num_speakers = {}
    
#             for num_speakers in range(2, 10+1):
#                 clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
#                 score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
#                 score_num_speakers[num_speakers] = score
#             best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
#             print(f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
#         else:
#             best_num_speaker = num_speakers
            
#         # Assign speaker label   
#         clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
#         labels = clustering.labels_
#         for i in range(len(segments)):
#             segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

#         # Make output
#         objects = {
#             'Start' : [],
#             'End': [],
#             'Speaker': [],
#             'Text': []
#         }
#         text = ''
#         for (i, segment) in enumerate(segments):
#             if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
#                 objects['Start'].append(str(convert_time(segment["start"])))
#                 objects['Speaker'].append(segment["speaker"])
#                 if i != 0:
#                     objects['End'].append(str(convert_time(segments[i - 1]["end"])))
#                     objects['Text'].append(text)
#                     text = ''
#             text += segment["text"] + ' '
#         objects['End'].append(str(convert_time(segments[i - 1]["end"])))
#         objects['Text'].append(text)
        
#         time_end = time.time()
#         time_diff = time_end - time_start
#         memory = psutil.virtual_memory()
#         gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
#         gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
#         gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
#         system_info = f"""
#         *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
#         *Processing time: {time_diff:.5} seconds.*
#         *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
#         """
#         filename = video_file_path.replace('/','')
#         filename = filename.replace('.','')
#         filename = filename.strip()
#         save_path = "transcript/transcript_result"+filename+"iris.csv"
#         df_results = pd.DataFrame(objects)
#         df_results.to_csv(save_path)
#         return df_results, system_info, save_path
    
#     except Exception as e:
#         raise RuntimeError("Error Running inference with local model", e)

