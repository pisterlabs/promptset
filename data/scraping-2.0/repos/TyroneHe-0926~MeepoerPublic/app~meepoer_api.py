from concurrent.futures import process
from crypt import methods
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request
import rq_dashboard
import numpy as np
import cv2
import praw
from util.extraction_tools import get_bbox, search_queue
from util.reddit_crawler import get_img_list_api
from configs.config import *
import flask_cors
from src.search import *
from util.tools import remove_dup_img, remove_filler_words, url_to_image
from ast import literal_eval
import openai
import tweepy
import requests

app = Flask(__name__)
app.config.from_object(rq_dashboard.default_settings)
app.config["RQ_DASHBOARD_REDIS_URL"] = RQ_DASHBOARD_REDIS_URL
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")

flask_cors.CORS(app)

@app.route("/get_bbox", methods=['POST'])
def bbox():
    storage_image = request.files["file"].read()
    npimg = np.fromstring(storage_image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    bboxes = get_bbox(img)
    result_arr = []
    bbox_arr = []
    for index, bbox in enumerate(bboxes):
        result_arr.append({
            "coord": [bbox[0], bbox[1], (bbox[2]-bbox[0]), (bbox[3]-bbox[1])],
            "label": str(index)
        })
        bbox_arr.append({
            "index": str(index),
            "bbox": bbox
        })
    
    return {"data":result_arr, "bboxes":bbox_arr}
    
@app.route("/image_to_embedding", methods=['POST'])
def to_embedding():
    storage_image = request.files["file"].read()
    embedding_type = request.form.get("embedding_type")
    page = int(request.form.get("page"))
    npimg = np.fromstring(storage_image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    data = []
    if embedding_type == 'person':
        person_bbox = [list(literal_eval(request.form.get("person_bbox")))]
        data = search_queue(img, bbox=person_bbox, page=page)
    if embedding_type == 'place':
        data = search_queue(img, bbox=None, page=page)
    
    return {"data": data}

@app.route("/text_img_search", methods=['POST'])
def text_img_search():
    search_text = remove_filler_words(request.form.get("search_text"))
    page = int(request.form.get("page"))
    print(f"Requesting at page {page}")

    twitter_d = TwitterData().search({"text": search_text}, page_size=3, page=page)['data']
    instagram_d = InstagramData().search({"text": search_text}, page_size=3, page=page)['data']
    reddit_d = RedditData().search({'text': search_text}, page_size=3, page=page)['data']

    combined_d = remove_dup_img(sorted(twitter_d+instagram_d+reddit_d, key=lambda d: d['score'], reverse=True))

    return {"data": combined_d}

@app.route("/gpt3_write", methods=['POST'])
def gpt3_write():
    openai.api_key = GPT3_API_SECRET

    text_length = {
        "long": "200 words",
        "regular": "50 words",
        "short": "20 words"
    }

    language = {
        "en": "English",
        "cn": "Chinese",
        "jp": "Japanese"
    }

    mood = request.form.get("mood") if request.form.get("mood") != "undefined" else None
    form = request.form.get("form") if request.form.get("form") != "undefined" else 'blog'
    length = request.form.get("length") if request.form.get("length") != "undefined" else 'regular'
    lan = request.form.get("lan") if request.form.get("lan") != "undefined" else 'en'
    topic = request.form.get("topic")

    prompt=f"\nHuman: Write a {text_length.get(length)} {mood} {form} about {topic} in {language.get(lan)}"

    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        temperature=0.9,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    result = response['choices'][0]['text'][2:]

    return {"data": result}

@app.route("/gpt3_rewrite", methods=['POST'])
def gpt3_rewrite():
    openai.api_key = GPT3_API_SECRET

    language = {
        "en": "English",
        "cn": "Chinese",
        "jp": "Japanese"
    }

    lan = request.form.get("lan") if request.form.get("lan") != "undefined" else 'en'
    text = request.form.get("text")

    if len(text) > 2049:
        return {"data": "Unable to create text"}

    prompt=f"\nHuman: Rewrite the sentence '{text}' in {language.get(lan)}"

    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        temperature=0.9,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    result = response['choices'][0]['text'][2:]

    return {"data": result}

@app.route("/parse_img_from_url", methods=['POST'])
def parse_img():
    def parse_tweet_url(url):
        auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
        auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
        crawler = tweepy.API(auth, wait_on_rate_limit=True)

        tweet_id = url.split("/")[-1].split("?")[0]
        tweet = crawler.search_tweets(q=tweet_id, tweet_mode="extended")[0]._json
        
        imgs = []
        try:
            for media_item in tweet["extended_entities"]["media"]:
                imgs.append(media_item["media_url_https"])
        except Exception as e:
            try:
                for media_item in tweet["retweeted_status"]["extended_entities"]["media"]:
                    imgs.append(media_item["media_url_https"])
            except Exception as e:
                try:
                    for media_item in tweet["quoted_status"]["extended_entities"]["media"]:
                        imgs.append(media_item["media_url_https"])
                except Exception as e:
                    for media_item in tweet["retweeted_status"]["quoted_status"]["extended_entities"]["media"]:
                        imgs.append(media_item["media_url_https"])
        
        return imgs
    
    def parse_ig_url(url):
        imgs = []
        img_url = url + 'media?size=m'
        resp = requests.get(img_url)
        imgs.append(resp.url)
        return imgs

    def parse_reddit_url(url):
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, 
                    client_secret=REDDIT_SECRET, 
                    user_agent="script by u/bben555")
        
        submiss = reddit.submission(url=url)

        if submiss.url.endswith(('.jpg', '.png', '.jpeg')):
            img_list = get_img_list_api(submiss, False)
        if "/gallery/" in submiss.url:
            img_list = get_img_list_api(submiss, True)
        
        return img_list

    url = request.form.get("url")
    source = request.form.get("source")
    
    if source == 'twitter':
        imgdata = parse_tweet_url(url)
    elif source == 'reddit':
        imgdata = parse_reddit_url(url)
    elif source == 'instagram':
        imgdata = parse_ig_url(url)

    return {"data": imgdata}

@app.route("/url_to_embedding", methods=['POST'])
def url_to_embedding():
    url = request.form.get("url")
    page = int(request.form.get("page"))
    img = url_to_image(url)

    data = search_queue(img, bbox=None, page=page)
    
    return {"data": data}


if __name__ == '__main__':
    port = 5000
    if len(sys.argv) > 1:
        port = sys.argv[1]

    app.run(host='0.0.0.0', port=port, debug=False)
