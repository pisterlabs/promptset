import openai
import requests
import os
openai.api_key = os.environ['KEY_API']

SENT = f"""raiva, alegria, medo, nojo, tristeza, satisfação, confiança, amor ou esperança"""

def get_music_lyric(artist_name, track_name):
    url = f'http://api.musixmatch.com/ws/1.1/matcher.lyrics.get'
    params = {
        'format': 'json',
        'q_artist': artist_name,
        'q_track': track_name,
        'apikey': os.environ['MUSIX']
    }

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200 and data['message']['header']['status_code'] == 200:
        lyrics = data['message']['body']['lyrics']['lyrics_body']
        lyrics = lyrics.replace("\n******* This Lyrics is NOT for Commercial use *******\n(1409623849948)", "") if type(lyrics) == str and "\n******* This Lyrics is NOT for Commercial use *******\n(1409623849948)" in lyrics else lyrics
        return lyrics
    
    return None

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_sentiment_by_lyrics(music):

    prompt = f"""
    Escolha entre {SENT} e associei a cada verso da música abaixo. Você deve escolher obrigatoriamente entre {SENT}. \
    E relacionar com cada verso da música abaixo. \
    Você deve retornar apenas o verso e o que foi escolhido entre {SENT} separado por hifen, um verso em cada linha \

    {music} \
    """

    response = get_completion(prompt)
    data = format_response(response)
    return {"data": data}

def get_sentiment_by_title(artist, track):
    lyric = get_music_lyric(artist, track)
    if not lyric:
        raise Exception("None lyric")
    else:
        response = get_sentiment_by_lyrics(lyric)
        return response

def format_response(res):
    arr = res.split("\n")
    formated_res = []
    for s in arr:
        splited = s.split("-")
        res = []
        i = 0
        for item in splited:
            if item != '':
                splited = item.split(" ")
                res2 = []
                for item in splited:
                    if item != '':
                        res2.append(item)
                if i == 1:
                    res.append(" ".join(res2).lower())
                else:
                    res.append(" ".join(res2))
                i += 1

        if len(res) == 2:
            formated_res.append(res)

    return formated_res