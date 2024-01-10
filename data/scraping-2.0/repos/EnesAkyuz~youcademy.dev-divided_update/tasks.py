# ~ lib.py

import json
import multiprocessing
import os
import time

import openai
import redis
import rq

from image_search_api import get_images
from gcp_text_to_speech import synthesize_text_with_audio_profile
from worker import conn

r = redis.Redis(host='localhost', port=6379, db=0)
q = rq.Queue(connection=conn)
openai.api_key = os.environ.get("OPENAI_API_KEY")

def qid2qck(qid): return f"prompt-{qid}"

def lang_id2natural_lang_lang(lid):
  return {
    "en": "English",
    "nl": "Dutch",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "te": "Telugu",
    "tr": "Turkish",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "ms": "Malay",
    "ur": "Urdu",
    "fa": "Persian",
    "mr": "Marathi",
    "da": "Danish",
  }.get(lid, "English")

def get_retry():
  return rq.Retry(max=10, interval=[1] * 7 + [3, 5, 10])


def generate_clip(text, ip, i, qid, language_id) -> dict: # instance path, index in slides, audio id, query id
  fn = get_audio_clip_path(ip, qid, i)
  language_code = {
    "en": "en-US",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "nl": "nl-NL",
    "it": "it-IT",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "zh": "zh-CN",
    "ar": "ar-AE",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "pa": "pa-IN",
    "te": "te-IN",
    "tr": "tr-TR",
    "id": "id-ID",
    "vi": "vi-VN",
    "th": "th-TH",
    "ms": "ms-MY",
    "ur": "ur-PK",
    "fa": "fa-IR",
    "mr": "mr-IN",
    "da": "da-DK",
  }.get(language_id, "en-US")
  synthesize_text_with_audio_profile(text, output=fn, language_code=language_code)
  return dict()

def generate_image(query) -> dict:
  image_url = get_images(query)[0]
  return dict(url=image_url)


MAX_TRIES = 5
def execute(task):
  print("exeucting task", task)
  try_i = 0
  while try_i < MAX_TRIES:
    try:
      func = task.get("func")
      ret = func(**task.get("kwargs"))
      return dict(**ret, status="ready")
    except Exception as e:
      print("Encountered exception", task, e)
    try_i += 1
    time.sleep(1.5 ** try_i) # exponential delay
  return dict(status="failed")


def generate_presentation(qid, query_text, user_prof, instance_path, language_id):
  language_name = lang_id2natural_lang_lang(language_id)  
  query = f'Write content for a slide deck about {query_text} for a listener who is a {user_prof}. Write the result in {language_name}. Return a JSON list with data about each slide. The list consists of ' \
              f'objects with a "text" key containing a paragraph ' \
              'that is spoken by the presenter, a "title" key to name the slide with low verbose, a "key_points" key which will contain a list of 3 key points (a couple of ' \
              'words) to be shown on the slide, and an "image_query" key that will contain the name of an image to illustrate each point with low verbose'
  response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
    {"role": "user", "content": query},
  ])

  qck = qid2qck(qid)
  presentation = r.get(qck)
  presentation = json.loads(presentation) # TODO: if this fails, we should retry the api call.
  presentation["original_query"] = query

  print("got response", response)

  response = response['choices'][0]['message']['content']
  presentation["original_response"] = response
  slides = json.loads(response)
  for i in range(len(slides)):
    slides[i]["audio_status"] = "pending"
    slides[i]["image_status"] = "pending"
    slides[i]["is_follow_up"] = False

  tasks = []
  for i, text in enumerate([slide["text"] for slide in slides]):
    tasks.append(dict(func=generate_clip, kwargs=dict(text=text, i=i, qid=qid, language_id=language_id, ip=instance_path)))
  for image_query in [slide["image_query"] for slide in slides]:
    tasks.append(dict(func=generate_image, kwargs=dict(query=image_query)))

  with multiprocessing.Pool(len(tasks)) as p:
    results = p.map(execute, tasks)

  print("results", results)

  assert len(results) == len(slides) * 2
  audio_rets = results[:len(slides)]
  image_rets = results[len(slides):len(slides)*2]

  for i, audio_ret in enumerate(audio_rets):
    slides[i]["audio_status"] = audio_ret.get("status")
  for i, image_ret in enumerate(image_rets):
    slides[i]["image_url"] = image_ret.get("url")
    slides[i]["image_status"] = image_ret.get("status")

  print("made slides", slides)

  presentation["slides"] = slides
  presentation["status"] = "completed"
  r.set(qid2qck(qid), json.dumps(presentation))

def answer_follow_up_question(qid, slide_idx, question, instance_path, language_id):
  language_name = lang_id2natural_lang_lang(language_id)  

  qck = qid2qck(qid)
  presentation = r.get(qck)
  presentation = json.loads(presentation)
  slide_title = presentation["slides"][slide_idx]["title"]

  response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
      {"role": "user", "content": presentation["original_query"]},
      {"role": "assistant", "content": presentation['original_response']},
      {"role": "user", "content": f'The listener has a question about the slide about {slide_title}. The listener says "{question}" Create a new slide, using the same structure as the other slides, to answer this question. Again answer in {language_name}. Return just the JSON dictionary for the new slide..'}])
  print("got response to follow up question:", response)
  response = response['choices'][0]['message']['content']
  print("got response to follow up question: ", response)
  new_slide = json.loads(response)
  try:
    new_slide = new_slide[0]
  except:
    pass

  new_slide["audio_status"] = "pending"
  new_slide["image_status"] = "pending"
  new_slide["is_follow_up"] = True

  new_slide_idx = slide_idx + 1

  presentation["slides"].insert(new_slide_idx, new_slide)
  # rename the audio files
  for i in reversed(range(new_slide_idx, len(presentation["slides"]) - 1)): # len -1 because we just added a new slide
    old_fn = get_audio_clip_path(instance_path, qid, i)
    new_fn = get_audio_clip_path(instance_path, qid, i+1)
    os.rename(old_fn, new_fn)
    print("renamed", old_fn, "to", new_fn)
  presentation["status"] = "completed"

  r.set(qck, json.dumps(presentation)) 

  q.enqueue_call(generate_clip, args=(new_slide["text"], instance_path, new_slide_idx, qid, language_id), retry=get_retry())
  q.enqueue_call(generate_image, args=(new_slide["image_query"], new_slide_idx, qid), retry=get_retry())

def get_audio_clip_path(ip, qid, i):
  return os.path.join(ip, "audio", f"{qid}-{i}.mp3")
