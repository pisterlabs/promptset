#!/usr/bin/env python3

import openai    # library for working with the OpenAI API
import re, os, sys, traceback, time
from datetime import datetime
from tqdm import tqdm
import config
from googletrans import Translator, LANGUAGES


# Retrieve configurations from the config file
youtube_list = config.YouTube_List
PREFIX = config.PREFIX
LANGUAGES = config.LANGUAGES

openai.organization = "YOUR_OPENAI_ORGANIZATION"
openai.organization = config.OpenAI_Organization
openai.api_key = "YOUR_OPENAI_API_KEY"
openai.api_key = config.OpenAI_Key

def rephrase_text(text, language="zh"):
    try:
        #print("-->" + text)
        if language=="zh":
            q = f"請幫我將下列文字更正錯字、加標點符號、轉成台灣的繁體中文，並且讓內容更通順:\n\n{text}\n\n 修正後文字:"
        else:
            q = f"Please rephrase the following text:\n{text}\n\nRevision:"

        rsp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Editor"},
                {"role": "user", "content": q}
            ]
        )

        summary = rsp.get("choices")[0]["message"]["content"].strip()
        #print("===>" + summary)
        return summary
    except Exception as ex:
        print(f"Exception type : {type(ex).__name__}")
        print(f"Exception message : {str(ex)}")
        print("Stack trace :")
        traceback.print_exc()

        time.sleep(1)
        return ""

def split_article(article, language="en", max_words=1000):
    word_count = 0   # word count of the current piece
    pieces = []      # list to store article pieces
    current_piece = ""

    if language=="zh":
        max_words = 1000
        lines = re.split(r"，", article)  # split article into lines at each period and space
    else:
        lines = re.split(r"\.\s", article)  # split article into lines at each period and space

    for line in lines:
        if language=="zh":
            line = line + "，"  # add period and space to end of line for grammatical correctness
            words_length = len(line)  # get length of words
        else:
            line = line + ". "  # add period and space to end of line for grammatical correctness
            words = line.split()  # split line into words
            words_length = len(words)  # get length of words list

        if ((word_count + words_length) > max_words):  # if word count exceeds max_words
            rephrased_text = ""
            while rephrased_text=="":
                rephrased_text = rephrase_text(current_piece, language)  # send current piece to rephrase_text function for modification
            pieces.append(rephrased_text)  # append modified piece to pieces list
            current_piece = line  # reset current piece to current line
            word_count = words_length  # reset word count to length of current line's words list
        else:
            current_piece += line  # add current line to current piece
            word_count += words_length  # increment word count by length of current line's words list

    pieces.append(current_piece)  # append last current piece to pieces list
    return pieces  # return list of article pieces

def summarize_text(text, language="zh"):
    if language=="zh":
        q = f"請依據下列的影片逐字稿內容，使用繁體中文進行摘要:\n{text}\n\n摘要:"
    else:
        q = f"Please summarize the following text:\n{text}\n\nSummary:"

    rsp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Editor"},
            {"role": "user", "content": q}
        ]
    )

    summary = rsp.get("choices")[0]["message"]["content"].strip()
    return summary

# iterate over the items in the youtube_list dictionary
for k, v in youtube_list.items():
    start_time = datetime.now()
    try:
        lang = "zh"
        msg = ""

        with open(f"{PREFIX}{k}.txt") as infile:
            lines = infile.read().splitlines()
            for line in lines:
                msg += line + "，"

        msgs = split_article(msg, lang)

        while len(msgs)>1:
            summary = ""
            for m in msgs:
                r = summarize_text(m, lang)
                summary += r
            msgs = split_article(summary, lang)

        with open(f"{PREFIX}{k}-summary.txt", "w") as out_file:
            out_file.write(msgs[0])
            out_file.close()

        for lang in LANGUAGES:
            translator = Translator()
            translated_text = translator.translate(text=msgs[0], dest=lang)
            with open(f"{PREFIX}{k}-summary_{lang}.txt", "w") as out_file:
                out_file.write(translated_text.text)
                out_file.close()

    except BaseException as ex:
        print(f"Exception type : {type(ex).__name__}")
        print(f"Exception message : {str(ex)}")
        print("Stack trace :")
        traceback.print_exc()

    end_time = datetime.now()
    delta_time = end_time - start_time
    print(os.path.basename(__file__) + "," + k + "," + v + "," + str(delta_time.total_seconds()))

