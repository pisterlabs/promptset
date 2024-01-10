# -*- coding: utf-8 -*-
import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEYY')
from generalGpt3 import GPT3

prompt_en =   """Intent classification is a classification problem that predicts the intent label \
and slot filling is a sequence labeling task that tags the input word sequence with the slot label. \
Here are some examples of joint intent classification and slot filling:

Query: Find me a movie by Steven Spielberg
Intent: find_movie
Slot: genre = movie, director = Steven Spielberg

Query: I want to book the next flight to London
Intent: book_tickets
Slot: transport = airplane, destination = London

Query: I want to go to a restaurant nearby that serves Indonesian food
Intent: find_restaurant
Slot: location = nearby, food = Indonesian

Query: I want to buy a new Dell pc
Intent: find_product
Slot: category = computer, brand = Dell

Query: I want to read a book by Jane Austen
Intent: read_book
Slot: author = Jane Austen

Query: The train leaves the station for Oslo at 8 o'clock
Intent: buy_train_ticket
Slot: transportation = train, destination = Oslo

Query: Find me an explanation of evolution for dummies
Intent: find_explanation
Slot: topic = evolution, audience = dummies

Query:"""

prompt_no =   """ Intensjonklassifisering er en klassifiseringsoppgave som bestemmer intensjonen i en setning, mens \
markering merker en setning med merkelapper. \
Under er noen eksempler på både intensjonklassifisering og markering:

Spørsmål: Finn en film av Steven Spielberg
Intensjon: finne_film
Merkelapper: sjanger = film, regissør = Steven Spielberg

Spørsmål: Jeg ønsker å booke neste fly til London
Intensjon: booke_billetter
Merkelapper: transport = fly, destinasjon = London

Spørsmål: Jeg vil dra på en restaurant i nærheten som severer indonesisk mat
Intensjon: finne_restaurant
Merkelapper: Sted = nærme, mat = indonesisk

Spørsmål: Jeg vil kjøpe en ny Dell pc
Intensjon: kjøpe_produkt
Merkelapper: kategori = datamaskin, merke = Dell

Spørsmål: Jeg vil lese en bok skrevet av Jane Austen
Intensjon: lese_bok
Merkelapper: forfatter = Jane Austen

Spørsmål: Toget forlater stasjonen og reiser mot Oslo klokken 20
Intensjon: kjøpe_togbilletter
Merkelapper: transport = tog, destinasjon = Oslo, klokkeslett = 20


Spørsmål: """
GPT3(prompt_no, "Spørsmål:", "Intensjon:", temperature=0.7,frequency_penalty=0.8,presence_penalty=0.6)
