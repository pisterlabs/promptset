# -*- coding: utf-8 -*-
# Copyright (C) 2023  HAL9000COM <f1226942353@icloud.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# %%

import os

import requests
import urllib.request


# %%
def video2audio(video_path: str, audio_path=None, bitrate="128k"):
    if audio_path is None:
        audio_path = video_path.split(".")[0] + ".webm"
    import ffmpeg

    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(
        stream,
        audio_path,
        codec="libopus",
        audio_bitrate=bitrate,
        ab=bitrate,
        af="pan=mono|c0=c0+c1",
        vn=None,
        sn=None,
    )
    ffmpeg.run(stream, overwrite_output=True)
    return audio_path


# %%
def audio2srt(
    audio_path: str,
    api_key: str,
    api="OpenAI",
    srt_path=None,
    en=False,
):
    if srt_path is None:
        srt_path = audio_path.split(".")[0] + ".srt"
    # check audio file size
    match api:
        case "OpenAI":
            import openai

            openai.api_key = api_key
            if os.path.getsize(audio_path) > 25 * 1024 * 1024:
                raise Exception(
                    "Audio file size is too large. Please use a smaller audio file."
                )
            audio_file = open(audio_path, "rb")
            openai.proxy = urllib.request.getproxies()["https"]
            if en:
                transcript = openai.Audio.translate(
                    "whisper-1", audio_file, response_format="srt"
                )
            else:
                transcript = openai.Audio.transcribe(
                    "whisper-1", audio_file, response_format="srt"
                )
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(transcript)  # type: ignore
        case "AutoSub-WebAPI":
            # Define the API endpoint URL
            url = api_key
            headers = {}
            audio_name = os.path.basename(audio_path)
            file = {
                "file": (audio_name, open(audio_path, "rb"), "audio/webm"),
            }
            settings = {"format": "srt"}
            response = requests.post(url, headers=headers, files=file, data=settings)
            if response.status_code == 200:
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(response.text)  # type: ignore
            else:
                raise Exception(
                    f"Request failed with status code {response.status_code}:"
                )
        case "Whisper-timestamped-WebAPI":
            # Define the API endpoint URL
            url = api_key
            headers = {}
            audio_name = os.path.basename(audio_path)
            file = {
                "file": (audio_name, open(audio_path, "rb"), "audio/webm"),
            }
            settings = {"format": "srt"}
            response = requests.post(url, headers=headers, files=file, data=settings)
            if response.status_code == 200:
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(response.text)  # type: ignore
            else:
                raise Exception(
                    f"Request failed with status code {response.status_code}:"
                )
        case _:
            raise Exception("Not supported transcribe API")
    return srt_path


def translate(text: list, target_lang: str, api_key: str, api="DeepL"):
    params = {"auth_key": api_key, "text": text, "target_lang": target_lang}
    match api:
        case "DeepL":
            request = requests.post(
                "https://api-free.deepl.com/v2/translate",
                data=params,
                proxies=urllib.request.getproxies(),
            )
            results = request.json()
            result_list = []
            for result in results["translations"]:
                result_list.append(result["text"])
            return result_list

        case "DeepLPro":
            request = requests.post(
                "https://api.deepl.com/v2/translate",
                data=params,
                proxies=urllib.request.getproxies(),
            )
            results = request.json()
            result_list = []
            for result in results["translations"]:
                result_list.append(result["text"])
            return result_list

        case "Google":
            import googletrans

            translator = googletrans.Translator()
            translations = translator.translate(text, dest=target_lang)
            result_list = []
            for translation in translations:
                result_list.append(translation.text)
            return result_list
        case _:
            raise Exception("Not supported translate API")


def translate_sub(
    sub_path: str,
    api_key: str,
    api: str,
    max_lines=20,
    target_lang="ZH",
    bilingual="Top",
):
    import pysrt

    subs = pysrt.open(sub_path)

    from itertools import zip_longest

    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    sub_list = list(grouper(subs, max_lines))

    for sub in sub_list:
        sub_text = []
        for s in sub:
            try:
                sub_text.append(s.text)
            except:
                pass
        translated = translate(sub_text, target_lang, api_key, api)

        for i, s in enumerate(sub):
            try:
                if bilingual == "Top":
                    s.text = translated[i] + "\n" + s.text
                elif bilingual == "Bottom":
                    s.text = s.text + "\n" + translated[i]
                elif bilingual == "None":
                    s.text = translated[i]
                else:
                    raise Exception("Not supported bilingual")
            except:
                pass  # ignore empty subtitle
    subs.save(sub_path, encoding="utf-8")
    return sub_path
