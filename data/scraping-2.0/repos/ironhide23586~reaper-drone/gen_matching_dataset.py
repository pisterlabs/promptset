"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""

import os
import pickle
from glob import glob
import ssl

import numpy as np

import utils

ssl._create_default_https_context = ssl._create_stdlib_context

from tqdm import tqdm
import cv2
from pytube import YouTube
from youtubesearchpython import VideosSearch
import openai

from dataset.retriever import HandCurated


MODE = 'train'
DATASET_DIR = utils.DOWNLOADED_DATASET_DIR
TOPIC = 'home tour indoor fpv scenic'

MAX_VIDEO_DURATION_IN_MINUTES = 10
MAX_SEARCH_VIDEOS = 50
NUM_GEN_PAIRS = 300
SEARCH_LISTS_DIR = 'scratchspace/youtube_video_search_lists'


def read_txt_file(fp):
    with open(fp, 'r') as f:
        links = f.readlines()
    links = [l.strip() for l in links]
    return links


if __name__ == '__main__':
    openai.api_key = utils.OPENAI_API_KEY
    openai.organization = "org-ZMb8qpLKF19bVrghd1crPe9y"
    os.makedirs(SEARCH_LISTS_DIR, exist_ok=True)

    link_fps = glob(SEARCH_LISTS_DIR + '/*_videos.txt')

    existing_links = list(np.hstack(list(map(read_txt_file, link_fps))))

    print('Asking ChatGPT for video title suggestions related to topic:', TOPIC)
    messages = [{"role": "system", "content":
                 "You are an intelligent assistant. You will be given a topic. Suggest relevant youtube search queries for videos relevant to the given topic"}]
    m = {'role': 'user', 'content': TOPIC}
    messages.append(m)
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    search_phrases_raw = chat.choices[0].message.content
    print('Raw Response -')
    print(search_phrases_raw)

    search_queries = list(map(lambda x: x.split(' \"')[-1].replace('\"', ''), search_phrases_raw.split('\n')[1:-1]))
    covered_titles = []

    condn = lambda l: len(l['duration'].split(':')) > 1 and len(l['duration'].split(':')) < 3 \
                      and int(l['duration'].split(':')[-2]) < MAX_VIDEO_DURATION_IN_MINUTES
    for video_search_phrase in search_queries:
        print('Searching YouTube with query:', video_search_phrase)
        search_tag = '_'.join([video_search_phrase.replace(' ', '-'), MODE])

        videos_search = VideosSearch(video_search_phrase, limit=10 * MAX_SEARCH_VIDEOS)
        search_results = videos_search.result()['result']
        links = [l['link'] for l in search_results if l['link'] not in existing_links and condn(l)]
        titles = [l['title'] for l in search_results if l['link'] not in existing_links and condn(l)]

        n_dropped = len(search_results) - len(links)
        if n_dropped > 0:
            print('Dropped', n_dropped, 'pre-existing videos')

        li = np.arange(len(links))
        np.random.shuffle(li)
        links = np.array(links)[li[:MAX_SEARCH_VIDEOS]]
        titles = np.array(titles)[li[:MAX_SEARCH_VIDEOS]]
        existing_links += list(links)

        print('Video Titles -')
        print(titles)

        link_fpath = SEARCH_LISTS_DIR + os.sep + search_tag + '_videos.txt'
        title_fpath = SEARCH_LISTS_DIR + os.sep + search_tag + '_videos-titles.txt'

        with open(link_fpath, 'w') as f:
            f.write('\n'.join(links))
        with open(title_fpath, 'w') as f:
            f.write('\n'.join(titles))

        out_dir = os.sep.join([DATASET_DIR, 'videos', MODE])
        os.makedirs(out_dir, exist_ok=True)

        i = 1
        local_video_fps = []
        for link in links:
            print('Downloading', link, i, '/', len(links))
            try:
                yt = YouTube(link)
                title = yt.title
                stream = yt.streams.filter(progressive=True, file_extension='mp4', res='720p')
                if len(stream) > 0:
                    out_fp = out_dir + os.sep + title + '.mp4'
                    if not os.path.exists(out_fp):
                        print('Downloading from', link)
                        out_fp = stream[0].download(out_dir)
                        print('Downloaded to', out_fp, '!')
                    else:
                        print(out_fp, 'exists..., skipping download')
                    local_video_fps.append(out_fp)
                else:
                    print('Failed :(')
            except Exception as e:
                print('Download aborted. Resaon - ', e)
            i += 1
        if len(local_video_fps) > 0:
            ds = HandCurated(local_video_fps, mode=MODE)
            dirname = '_'.join([search_tag.replace('_' + MODE, ''), ds.tag])
            viz_dir = 'scratchspace/gt_viz_images/' + dirname
            data_dir = 'scratchspace/gt_data/' + dirname
            os.makedirs(viz_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            print('Generating dataset...', dirname)
            for i in tqdm(range(NUM_GEN_PAIRS)):
                x, y, fm, v = ds.sample_image_pair()
                fn = '_'.join([str(i), str(y[0].shape[0]), str(fm[1][1] - fm[0][1]), fm[0][0], str(fm[0][1]), str(fm[1][1])])
                fp = viz_dir + os.sep + fn + '.jpg'
                cv2.imwrite(fp, v)
                payload = (x, y[0], fm)
                fp = data_dir + os.sep + fn + '.pickle'
                with open(fp, 'wb') as f:
                    pickle.dump(payload, f)
        else:
            print('No videos found, moving on....')
