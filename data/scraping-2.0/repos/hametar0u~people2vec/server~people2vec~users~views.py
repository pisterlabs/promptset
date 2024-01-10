from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound
from rest_framework.decorators import api_view
import json
from django.views.decorators.csrf import csrf_exempt
from users.models import User

import io
import os
import random
from urllib import request

import cohere
import numpy as np
from PIL import Image
from scipy import linalg
import pandas as pd
import re
import requests
import torch
from torch import nn
from torchvision import models, transforms
from tqdm.auto import tqdm
from users.models import User


N = 1000
N_vid = 100


# TEXT2VEC

co = cohere.Client("gr3eALiN829VBG4rWgT7YrCFZTNZlEoHDIYLdpaP")
embed_dim = {"small": 1024, "large": 4096, "multilingual-22-12": 768}


def text2vec(texts, model="multilingual-22-12"):
    assert model in embed_dim.keys()

    embeds = co.embed(texts=texts, model=model)
    embeds = np.asarray(embeds.embeddings, dtype=np.float32)
    return embeds


# IMAGE2VEC

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

inception_v3 = models.inception_v3(weights="IMAGENET1K_V1")
inception_v3.fc = nn.Identity()
inception_v3.to(device)
inception_v3.eval()

to_tensor = transforms.ToTensor()

pil_preprocess = lambda image: np.asarray(image.resize((1280, 720), Image.LANCZOS))
transform = transforms.Compose([
    transforms.Resize((299, 299), antialias=True),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


@torch.no_grad()
def image2vec(images):  # Expects image as NumPy array, uint8, [0, 255]
    images = torch.from_numpy(images).to(torch.float32).moveaxis(-1, -3) / 255
    images = transform(images).to(device)
    features = inception_v3(images).detach().cpu().numpy()
    return features


def link2vec(links, N=None):
    N = len(links) // 100 if N is None else N
    images = []
    progress = tqdm(total=N, desc="link2vec")
    i = 0
    while i < N:
        try:
            with request.urlopen(random.choice(links)) as url:
                file = io.BytesIO(url.read())
        except:
            continue
        image = pil_preprocess(Image.open(file))
        images.append(image)
        i += 1
        progress.update(1)
    images = np.stack(images, axis=0)
    features = image2vec(images)
    return features


# EVAL

def cos_sim(vec_1, vec_2):
    assert vec_1.shape == vec_2.shape
    score = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    return score
  
  
def matrix_sqrt(x):
    x = linalg.sqrtm(x, disp=False)[0].real
    return x


def feature_statistics(f):
    mu = np.mean(f, axis=0)
    sigma = np.cov(f, rowvar=False)
    return mu, sigma


def frechet_distance(mu_1, sigma_1, mu_2, sigma_2, epsilon=1e-7):
    assert mu_1.shape == mu_2.shape
    assert sigma_1.shape == sigma_2.shape

    sse = np.sum(np.square(mu_1 - mu_2))
    covariance = matrix_sqrt(sigma_1 @ sigma_2)
        
    if np.isinf(covariance).any():
        I = np.eye(sigma_1.shape[0])
        covariance = matrix_sqrt(sigma_1 @ sigma_2 + epsilon * I)

    fid = sse + np.trace(sigma_1) + np.trace(sigma_2) - 2 * np.trace(covariance)
    return fid


def fid(f_1, f_2):
    # It is recommended, however, to compute feature_statistics only once and store the resulting
    # mu and sigma instead of recalculating it every time with this function
    assert f_1.shape == f_2.shape
    mu_1, sigma_1 = feature_statistics(f_1)
    mu_2, sigma_2 = feature_statistics(f_2)
    fid = frechet_distance(mu_1, sigma_1, mu_2, sigma_2)
    return fid 


# Create your views here.

def index(request):
    return HttpResponse("hello world")


def get_user_dir(user):  # gets the tsv of the user data based on their username
    root_dir = "data/raw_data"
    user_dir = f"{root_dir}/{user}.tsv"
    return user_dir


def get_features_dir(user, match_type):
    root_dir = "data/features"
    features_dir = f"{root_dir}/{match_type}_{user}_features.npy"
    return features_dir


def get_stats_dir(user, match_type):
    root_dir = "../stats"
    stats_dir = (f"{root_dir}/{match_type}_{user}_stats_mu.npy",
                 f"{root_dir}/{match_type}_{user}_stats_sigma.npy")
    return stats_dir


def calculate_feature_statistics(user, match_type):
    history = pd.read_csv(get_user_dir(user), sep="\t")[:N]  # read in tsv of user data
    data = history[match_type].tolist()  # get title/thumbnail as a list, [:N] select recent N
    
    if match_type == "title":  # features are 768-dimensional, so features is N x 768
        features = text2vec(data)
    elif match_type == "thumbnail":  # N x 2048 (NumPy array)
        features = link2vec(data, N=N_vid)
    mu, sigma = feature_statistics(features)  # computes the mu and sigma of features
    # mu 1 x (768 or 2048), sigma (768 or 2048) x (768 or 2048)
    
    np.save(get_features_dir(user, match_type), features)
    np.save(get_stats_dir(user, match_type)[0], mu)
    np.save(get_stats_dir(user, match_type)[1], sigma)


def address_to_coordinates(address, api_key):
    """
    Converts an address to a latitude and longitude
    :param address:
    :return:
    """
    try:
        api_response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address='+address+'&key='+str(api_key))
        read_response = api_response.json()
        # address = read_response['results'][0]['formatted_address']
        lat = read_response['results'][0]['geometry']['location']['lat']
        lng = read_response['results'][0]['geometry']['location']['lng']
        # return {'formatted_address':address, 'lat':lat, 'lng':lng}
        return (lat, lng)
    except:
        return (None, None)


# Create your views here.
def index(request):
    return HttpResponse("hello world")

@api_view(['GET'])
def get_user_data(request):
    username = request.GET.get('username','')
    try:
        user = User.objects.get(username=username)
    except:
        return HttpResponseNotFound("user not found")
    
    else:
        return HttpResponse("logged in", status_code=200)
    
    
def direct_html_parser(html, username):
    # with open(html_path, 'r', encoding='utf-8') as f:
    #     html = f.read()
    results = re.findall('content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1.*?</div>', html)
    data = []
    for line in results:
        try:
            link, title, channel, time = re.findall('(https://www\.youtube\.com/watch.*?)>(.*?)<.*?https://www.youtube.com/channel.*?">(.*?)</a><br>(.*?)</div>', line)[0]
            video_id = re.findall('/watch\?v=(.*?)"$', link)[0]
            thumbnail = f'https://img.youtube.com/vi/{video_id}/0.jpg'
            screenshot_1 = f'https://img.youtube.com/vi/{video_id}/1.jpg'
            screenshot_2 = f'https://img.youtube.com/vi/{video_id}/2.jpg'
            screenshot_3 = f'https://img.youtube.com/vi/{video_id}/3.jpg'
            data.append([title, channel, time, link, video_id, thumbnail, screenshot_1, screenshot_2, screenshot_3])        
        except:
            pass
    data = pd.DataFrame(data, columns = ["title", "channel", "time", "link", 'video_id', 'thumbnail', 'screenshot_1', 'screenshot_2', 'screenshot_3'])
    data.to_csv(f"./data/raw_data/{username}.tsv", index=None, sep='\t')

@csrf_exempt
@api_view(['POST'])
def signup(request):
    # do some parsing shit
    html = request.data.get('html', '')
    username = request.data.get('username', '')
    try:
        direct_html_parser(html=html, username=username)  # save to tsv
        calculate_feature_statistics(username, "title")
        calculate_feature_statistics(username, "thumbnail")
        user = User(username = username, 
                    password = request.password, 
                    email = request.email, 
                    phone = request.phone,
                    description = request.description
                    )
        
        user.save()

        return HttpResponse("account created")
    except:
        return HttpResponse("Sum Ting Wong", status_code=500)
    
    
@csrf_exempt
def get_FID_scores(request):
    user_1 = request.GET.get("user_1", "")
    user_2 = request.GET.get("user_2", "")
    match_type = request.GET.get("match_type", "")
    
    stats_dir_1 = get_stats_dir(user_1.username, match_type)
    stats_dir_2 = get_stats_dir(user_2.username, match_type)
    
    mu_1 = np.load(stats_dir_1[0])
    sigma_1 = np.load(stats_dir_1[1])
    mu_2 = np.load(stats_dir_2[0])
    sigma_2 = np.load(stats_dir_2[1])
    
    fid = frechet_distance(mu_1, sigma_1, mu_2, sigma_2)
    
    return HttpResponse(fid)
