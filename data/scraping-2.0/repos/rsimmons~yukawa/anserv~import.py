import os
import json
import time
import argparse
import sqlite3
from urllib.parse import urlparse
from collections import Counter
from io import StringIO
import hashlib

import requests
import yt_dlp
import openai
from sudachipy import tokenizer, dictionary
import webvtt

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest='command')

parser_add_yt_source = subparsers.add_parser('add_yt_source')
parser_add_yt_source.add_argument('--url', required=True)
parser_add_yt_source.add_argument('--duration_max', type=int)

parser_update_yt_source = subparsers.add_parser('update_yt_source')
parser_update_yt_source.add_argument('--source_id', type=int, required=True)

parser_import_frags = subparsers.add_parser('import_frags')
parser_import_frags.add_argument('--number', type=int, required=True)

args = parser.parse_args()

conn = sqlite3.connect('anserv.db', isolation_level=None)
c = conn.cursor()

sudachi_tokenizer_obj = dictionary.Dictionary().create()

# algorithms:
# - ja1 - sudachi 0.6.7, dict core 20230927
def analyze_ja(text):
    normal_freq = Counter()
    morphemes = sudachi_tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.B)

    for m in morphemes:
        if not (m.part_of_speech()[0] in ['補助記号', '空白']):
            normal = m.normalized_form()
            normal_freq[normal] += 1

    return ('ja1', dict(normal_freq)) # convert to dict to be json-able

def format_yt_video_url(id):
    return f'https://www.youtube.com/watch?v={id}'

def add_yt_thumbnail(thumbs):
    thumb = thumbs[-1] # this seems to be the best thumbnail for both playlists and channels, which have different thumbnail sets
    image_url = thumb['url']
    parsed = urlparse(image_url)
    image_data = requests.get(image_url).content

    image_md5 = hashlib.md5(image_data).hexdigest()
    c.execute('INSERT INTO image (md5, data) VALUES (?, ?)', (image_md5, image_data))
    return c.lastrowid

def add_yt_video(info):
    id = info['id']
    url = info['url']

    YT_AUDIO_DIR = 'yt-audio'
    # ensure yt-audio subdir exists
    os.makedirs(YT_AUDIO_DIR, exist_ok=True)

    with yt_dlp.YoutubeDL({
        'format': 'bestaudio',
        'outtmpl': 'yt-audio/%(id)s.%(ext)s',
    }) as ydl:
        error_code = ydl.download([url])

    if error_code:
        raise Exception(f'error downloading {url}')

    # find audio output file by looking in output dir (can't be sure of extension)
    audio_fn = None
    for fn in os.listdir(YT_AUDIO_DIR):
        if fn.startswith(id):
            audio_fn = os.path.join(YT_AUDIO_DIR, fn)
            break
    if audio_fn is None:
        raise Exception(f'audio file not found for {url}')
    audio_ext = os.path.splitext(audio_fn)[1][1:]
    assert audio_ext in ['m4a', 'webm', 'mp3']

    # get audio data
    with open(audio_fn, 'rb') as audio_file:
        audio_data = audio_file.read()

    duration = info['duration']
    assert duration == int(duration)

    # do speech recognition
    openai.api_key = os.getenv('OPENAI_API_KEY')

    print(f'transcribing audio {audio_fn} for video {url}')
    with open(audio_fn, 'rb') as audio_file:
        transcript = openai.Audio.transcribe('whisper-1', audio_file, response_format='vtt')

    # do analysis
    captions = []
    for caption in webvtt.read_buffer(StringIO(transcript)):
        captions.append(caption.text)
    transcript_plain = '\n'.join(captions)
    (an_algo, an) = analyze_ja(transcript_plain)
    analysis_json = json.dumps({an_algo: an}, ensure_ascii=False, sort_keys=True)

    current_time = int(time.time())

    # update the database
    c.execute('BEGIN')

    # insert image row
    print(f'inserting image row for video {url}')
    image_id = add_yt_thumbnail(info['thumbnails'])

    # insert audio row
    print(f'inserting audio row for video {url}')
    audio_md5 = hashlib.md5(audio_data).hexdigest()
    c.execute('INSERT INTO audio (extension, md5, data) VALUES (?, ?, ?)', (audio_ext, audio_md5, audio_data))
    audio_id = c.lastrowid

    # insert a new piece row
    print(f'inserting piece row for video {url}')
    c.execute('INSERT INTO piece (url, kind, title, image_id, audio_id, duration, stt_method, text_format, text, analysis, time_fetched, time_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (url, 'video', info['title'], image_id, audio_id, duration, 'openai-api-whisper-1', 'vtt', transcript, analysis_json, current_time, current_time))
    piece_id = c.lastrowid

    conn.commit()

    return piece_id

def add_yt_source(url, filter_obj):
    # TODO: ensure url is canonical url for playlist or channel videos tab

    with yt_dlp.YoutubeDL({
        'extract_flat': 'in_playlist',
        'playlistend': 0,
    }) as ydl:
        info = ydl.extract_info(url, download=False)

    c.execute('BEGIN')

    image_id = add_yt_thumbnail(info['thumbnails'])

    filter_json = json.dumps(filter_obj, ensure_ascii=False, sort_keys=True) if filter_obj else None
    c.execute('INSERT INTO source (url, filter, kind, title, image_id, time_updated) VALUES (?, ?, "video", ?, ?, NULL)', (url, filter_json, info['title'], image_id))
    source_id = c.lastrowid

    conn.commit()

    _update_yt_source(source_id, url, filter_obj)

def update_yt_source(source_id):
    c.execute('SELECT url, filter FROM source WHERE id = ?', (source_id,))
    source_row = c.fetchone()
    if source_row is None:
        raise Exception(f'source id {source_id} does not exist')

    url = source_row[0]
    filter_obj = json.loads(source_row[1])

    _update_yt_source(source_id, url, filter_obj)

def _update_yt_source(source_id, url, filter_obj):
    vid_infos = []
    with yt_dlp.YoutubeDL({
        'extract_flat': 'in_playlist',
        'playlistend': 200,
    }) as ydl:
        info = ydl.extract_info(url, download=False)
        # print(json.dumps(info, indent=2, ensure_ascii=False))
        for vid_info in info['entries']:
            if vid_info['_type'] != 'url':
                continue
            if vid_info['duration'] is None: # this seems to skip unviewable videos
                continue
            if vid_info['availability'] == 'subscriber_only':
                continue

            # apply filter
            if filter_obj and 'duration_max' in filter_obj:
                if vid_info['duration'] > filter_obj['duration_max']:
                    print(f'skipping video {vid_info["url"]} because duration {vid_info["duration"]} > {filter_obj["duration_max"]}')
                    continue

            vid_infos.append(vid_info)
    print(f'listed {len(vid_infos)} videos from {url}')

    with yt_dlp.YoutubeDL({
        'format': 'bestaudio'
    }) as ydl:
        for vid_info in vid_infos:
            vid_url = format_yt_video_url(vid_info['id']) # to ensure it's canonical
            assert vid_url == vid_info['url']

            # check if we already have a piece row for this video url
            c.execute('SELECT id FROM piece WHERE url = ?', (vid_url,))
            piece_row = c.fetchone()

            if piece_row is not None:
                # we already have a piece row for this video, so continue
                piece_id = piece_row[0]
            else:
                print(f'adding video {vid_url}')
                piece_id = add_yt_video(vid_info)

                # insert piece_source row
                print(f'inserting piece_source row for video {vid_url} source {url}')
                c.execute('BEGIN')
                c.execute('INSERT INTO piece_source (piece_id, source_id) VALUES (?, ?)', (piece_id, source_id))
                conn.commit()

    # TODO: set time_updated

if args.command == 'add_yt_source':
    filter_obj = {}
    if args.duration_max is not None:
        filter_obj['duration_max'] = args.duration_max
    filter_obj = filter_obj or None

    add_yt_source(args.url, filter_obj)
elif args.command == 'update_yt_source':
    update_yt_source(args.source_id)
else:
    raise Exception('invalid command')
