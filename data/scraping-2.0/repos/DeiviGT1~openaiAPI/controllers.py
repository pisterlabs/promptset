from flask import render_template, request, redirect, session, Blueprint
from spotify import app_Authorization, search_song, user_Authorization
from openaiapi import generar_respuesta
from dotenv import load_dotenv



mod = Blueprint('controllers', __name__, url_prefix='')

#renders the web page
@mod.route("/")
def home():
    return render_template("index.html")

@mod.route("/login", methods=["POST","GET"])
def login():
    auth_url = app_Authorization()
    session["spotify"] = auth_url

    song = request.form["song"]
    session["song"] = song
    session["artist"] = request.form["artist"]
    return redirect(auth_url)

@mod.route("/callback")
def callback():
    
    header = user_Authorization()
    session["user"] = header
    return redirect("/song")

@mod.route("/song", methods=["POST","GET"])
def get_input():
    header = session.get("user")
    real_songs = []
    song = session.get("song")
    artist = session.get("artist")
    playlists = generar_respuesta(song, artist)

    lista_nueva = []
    for elemento in playlists:
        elemento_nuevo = elemento.replace("\"", "").replace("\u00f1", "n").replace("\u00e1", "a").replace("\u00e9", "e").replace("\u00ed", "i").replace("\u00f3", "o").replace("\u00fa", "u").replace("¿", "").replace("?", "").replace("¡", "").replace("!", "").replace(".", "").replace(",", "").replace("  ", " ").strip()
        lista_nueva.append(elemento_nuevo)
    
    for song in lista_nueva:
        result = search_song(header = header, song_name = song)
        if len(result["tracks"]["items"]) > 0:
            track_id = result["tracks"]["items"][0]["id"]
            track_image = result["tracks"]["items"][0]["album"]["images"][0]["url"]
            track_name = result["tracks"]["items"][0]["name"]
            track_artists = result["tracks"]["items"][0]["artists"][0]["name"]
            track_url = result["tracks"]["items"][0]["external_urls"]["spotify"]
            
            real_songs.append({"name": track_name, "artist":track_artists , "url": track_url, "image": track_image})

    return render_template("songs.html", songs = real_songs)
