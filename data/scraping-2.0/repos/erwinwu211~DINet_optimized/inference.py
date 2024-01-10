import concurrent.futures
import glob

# from utils.deep_speech import DeepSpeech
import logging
import os
import random
import subprocess
import time
from collections import OrderedDict
from timeit import default_timer

import cv2
import numpy as np
import torch
import sys
from config.config import DINetInferenceOptions
from models.DINet import DINet
from utils.data_processing import compute_crop_radius, load_landmark_openface
from utils.wav2vec import Wav2VecFeatureExtractor
from utils.wav2vecDS import Wav2vecDS
from pathlib import Path

import pygame as pg
from pygame._sdl2 import (
    get_audio_device_names,
    AudioDevice,
    AUDIO_S16,
    AUDIO_ALLOW_FORMAT_CHANGE,
)


from pathlib import Path
from openai import OpenAI
from a2m import A2M


class State:
    def __init__(self):
        self.recording = False
        self.sound_chunks = []
        self.audio = None


def setup_audio(state: State):
    pg.mixer.pre_init(44100, -16, 1, 2048)
    pg.init()

    names = get_audio_device_names(True)
    # Add a flag to control recording

    """This is called in the sound thread."""

    def callback(audiodevice, audiomemoryview):
        if state.recording:
            state.sound_chunks.append(bytes(audiomemoryview))

    # set_post_mix(callback)

    state.audio = AudioDevice(
        devicename=names[0],
        iscapture=True,
        frequency=44100 * 2,
        audioformat=AUDIO_S16,
        numchannels=1,
        chunksize=2048,
        allowed_changes=AUDIO_ALLOW_FORMAT_CHANGE,
        callback=callback,
    )
    state.audio.pause(0)


def is_speech_start(sound_chunks, threshold=1):
    return sound_chunks[-1] >= threshold


def setup_display():
    pg.init()
    WIDTH = 500
    HEIGHT = 500

    windowSurface = pg.display.set_mode((WIDTH, HEIGHT), 32)
    return windowSurface


# Frames extraction took 29.91 sec
def extract_frames_from_video(video_path, save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print(
            "warning: the input video is not 25 fps, it would be better to trans it to 25 fps!"
        )
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))

    os.makedirs(save_dir, exist_ok=True)
    # Construct the ffmpeg command
    ffmpeg_command = ["ffmpeg", "-i", video_path, os.path.join(save_dir, "%06d.png")]

    # Run the ffmpeg command
    subprocess.run(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )

    return frame_width, frame_height

 

def convert_opencv_img_to_pygame(opencv_image):
    """
    OpenCVの画像をPygame用に変換.

    see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
    """
    opencv_image = opencv_image[:,:,::-1]  # OpenCVはBGR、pygameはRGBなので変換してやる必要がある。
    shape = opencv_image.shape[1::-1]  # OpenCVは(高さ, 幅, 色数)、pygameは(幅, 高さ)なのでこれも変換。
    pygame_image = pg.image.frombuffer(opencv_image.tostring(), shape, 'RGB')

    return pygame_image

def device(data_root):

    state = State()
    windowSurface = setup_display()
    #setup_audio(state)
    clock = pg.time.Clock()
    vid_name = "/out2/"
    ori_vid_name = "/out/"
    video_path = data_root+ori_vid_name
    img_path = data_root+vid_name


    # client = OpenAI()
    # inp = input("Input the Questions: ")
    # inp = inp + " Please answer under 20 words, and start with 没问题"
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "system", "content": "You are a helpful assistant."},
    #               {"role": "user", "content": inp}]
    # )
    # assistant_res = response.choices[0].message.content

    # print(assistant_res)
    # speech_file_path = data_root+"/speech.mp3"
    
    # response = client.audio.speech.create(
    # model="tts-1",
    # voice="nova",
    # input=assistant_res
    # )
    # response.stream_to_file(speech_file_path)

    audio_path = data_root+"/speech.mp3"

    image_list = sorted(glob.glob(img_path+"*.png"))
    ori_vid_list = sorted(glob.glob(video_path+"*.png"))
    #model, pad_length, video_size, opt, ds_feature_padding,ref_img_tensor, resize_w,resize_h, res_video_landmark_data_pad, res_video_frame_path_list_pad = a2m_preprocess(img_path, image_list, audio_path)
    a2m = A2M()
    a2m.load(img_path, image_list, audio_path)
    f = 0
    running = True
    print("generated " + str(a2m.pad_length) + " frames")
    pg.mixer.music.load(audio_path)
    pg.mixer.music.play()
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

            elif event.type == pg.KEYDOWN:
                # Quit when the user presses the ESC key
                if event.key == pg.K_ESCAPE:
                    running = False
                # Start recording when SPACE key is pressed
                elif event.key == pg.K_SPACE:
                    state.sound_chunks = []
                    state.recording = True

            elif event.type == pg.KEYUP:
                # Stop recording when SPACE key is released
                if event.key == pg.K_SPACE:
                    state.recording = False
                    sound_data = pg.mixer.Sound(buffer=b"".join(state.sound_chunks))
                    sound = pg.mixer.Sound(buffer=sound_data)
                    sound.play()
        if f < a2m.pad_length - 2:
            img = a2m.gen_frame(f)  
            #a2m(model, f, video_size, opt, ds_feature_padding,ref_img_tensor, resize_w,resize_h, res_video_landmark_data_pad, res_video_frame_path_list_pad)
        else:
            img = cv2.imread(ori_vid_list[f-3])[200:700,300:800]
        #ori_img = cv2.imread(ori_vid_list[f])
        #ori_img[200:700,300:800] = img
        #ori_img = cv2.resize(ori_img, dsize=None, fx=0.5, fy=0.5)
        #ori_vid_list = cv2.resize()
        img = convert_opencv_img_to_pygame(img)

        windowSurface.blit(img, (0, 0))
        pg.display.flip()
        f += 1
        f = f % len(image_list)
        clock.tick(25)
    pg.display.quit()
    pg.quit()
    exit()

if __name__ == "__main__":
    device("asserts/examples/")
    #a2m()
