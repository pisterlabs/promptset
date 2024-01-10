from flask import Flask, redirect, render_template, request, url_for
import pandas as pd
import sqlalchemy as db
import requests
import openai
import json
import os
from dotenv.main import load_dotenv

load_dotenv()
favorite_language = os.environ['LANGUAGE']

#openai.api_key = 'API_KEY_OPENAI'

def generate_art():
    openai.api_key = "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"
    url = "https://pokeapi.co/api/v2/pokemon/"
    error = "Invalid Pokemon name!"
    pokemon_dict = {}

    print(r"""
 _______         __             _______   _          
|_   __ \       [  |  _        |_   __ \ (_)         
  | |__) | .--.  | | / ] .---.   | |__) |__   .---.  
  |  ___// .'`\ \| '' < / /__\\  |  ___/[  | / /'`\] 
 _| |_   | \__. || |`\ \| \__., _| |_    | | | \__.  
|_____|   '.__.'[__|  \_]'.__.'|_____|  [___]'.___.'                                                          
    """)
    print("Generate pokemon art using openAI's API.")
    print("Type 'exit' to exit the program, or 'show' to access all previous URLs.")
    while True:
        pokemon_name = input("Please enter the name of a Pokemon, 'exit', 'show', OR 'usage': ").lower()
        if len(pokemon_name) == 0:
            print(error + " Name cannot be empty.")
            continue
        elif pokemon_name == "usage":
            print("Generate pokemon art using openAI's API.")
            print("Type 'exit' to exit the program, or 'show' to access all previous URLs.")
            continue
        elif pokemon_name == "exit":
            return "Program successfully exited."
        elif pokemon_name == "show":
            if len(pokemon_dict) == 0:
                print("No previous URLs found.")
                continue
            dataframe = pd.DataFrame.from_dict(pokemon_dict, orient='index')
            engine = db.create_engine('sqlite:///data_base_name.db')
            dataframe.to_sql('table_name', con=engine, if_exists='replace', index=True)
            with engine.connect() as connection:
                connection.execute(db.text('ALTER TABLE table_name RENAME COLUMN "index" TO pokemon;'))
                connection.execute(db.text('ALTER TABLE table_name RENAME COLUMN "0" TO urls;'))
                query_result = connection.execute(db.text("SELECT pokemon, urls FROM table_name;")).fetchall()
                print(pd.DataFrame(query_result))
            find_pokemon = input("Type a pokemon name to access previous entries, or type 'return': ").lower()
            if find_pokemon == "return":
                continue
            try:
                found_pokemon = pokemon_dict[find_pokemon] 
            except:
                print(error + " Pokemon " + find_pokemon + " not found within the dictionary." )
                continue
            else:
                with engine.connect() as connection:
                    query_result = connection.execute(db.text('SELECT urls FROM table_name WHERE pokemon="' + find_pokemon + '";')).fetchall()
                    #pd.set_option('display.max_colwidth', 0)
                    pd.set_option('display.min_colwidth', 0)
                    print(pd.DataFrame(query_result))
                    #pd.reset_option('display.max_colwidth')
                    pd.reset_option('display.min_colwidth')
                continue
        try:
            response = requests.get(url + pokemon_name).json()
        except:
            print(error + " Pokemon " + pokemon_name + " not found." )
        else:
            desc = "Generate a fully colored, realistic art piece with a colorful background based on the following:\n\nPokemon: " + pokemon_name + "\nDescription:\n"
            desc += requests.get(response["species"]["url"]).json()["flavor_text_entries"][0]["flavor_text"]

            response = openai.Image.create(
            prompt=desc,
            n=1,
            size="1024x1024"
            )
            image_url = response['data'][0]['url']

            print("The following URL contains the generated artwork:")
            print(image_url)

            if pokemon_name in pokemon_dict.keys():
                pokemon_dict[pokemon_name] += image_url
            else:
                pokemon_dict[pokemon_name] = [image_url]

print(generate_art())