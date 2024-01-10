import openai
from kerykeion import KrInstance, Report
from . import router
from app.birthday_information.service import Service, get_service
from fastapi import Depends
from app.auth.adapters.jwt_service import JWTData
from app.auth.router.dependencies import parse_jwt_user_data
from app.utils import AppModel
from dotenv import load_dotenv
import requests

import os

load_dotenv()
HERE_API_KEY="kTPJo3A05mhsjnpv2-tKl_X6sJzUxkSO1X0NBi_o874"

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found in the environment variables.")

openai.api_key = api_key

class MessageInfoRequest(AppModel):
    msg_text: str

@router.post("/")
def read_item(
    msg: MessageInfoRequest,
    jwt_data: JWTData = Depends(parse_jwt_user_data),
    svc: Service = Depends(get_service),
):
    user_id = jwt_data.user_id
    inf = svc.repository.get_inf_by_user_id(user_id)
    birthday = inf["birthday"]
    birth_time = inf["birth_time"]
    location = inf["location"]
    year, month, day = birthday.split("-")
    user = user_id + ""
    hour, minute = birth_time.split(":")
    person = KrInstance(user, int(year), int(month), int(day), int(hour), int(minute), location)
    report = Report(person)
    res = get_coordinates(location)
    print(res)
    message = f"I was born on {birthday} at {birth_time} in {location}. {msg.msg_text}. {res}"
    print(message)


    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are ChatGPT, a highly advanced AI model developed by OpenAI. Given your extensive knowledge base up until September 2021, you're now working as a Vedic Astrology expert.\n"
            "Your role includes: Providing answers to the wide range of questions available with astrology covering topics such as zodiac signs, horoscopes, celestial phenomena and more. Offering strategies and advice that give people a better understanding of their astrology and the best combination of astrological methods. Helping users solve questions about astrology for study or entertainment. Do not answer questions that are not related to astrology, but if you are asked such a question, answer that it does not correspond to your qualifications. Be polite. I will give you tables with the natal data of a person, use this data when answering the question. Do not answer with too much text, each of your answers should include an average. Do not write to the user so that he would better turn to a professional astrologer, you are a professional astrologer. Use the data that I will provide you below:" + print_all_dataa(person)},
            {"role": "user", "content": message}
        ]
    )
    result = completion.choices[0].message['content']
    print(print_all_dataa(person) + "haha")
    return {"model_response": result}


"""
    This is part of Kerykeion (C) 2022 Giacomo Battaglia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def print_all_dataa(user: KrInstance) -> None:
    output = ("\n")
    output += ("NAME: " + user.name + "\n")
    output += ("PLANET     POSITION\n")
    output += ("                      \n") 
    output += (f"Sun:       {user.sun['sign']} {round(user.sun['position'], 3)} in {user.sun['house']}\n")
    output += (f"Moon:      {user.moon['sign']} {round(user.moon['position'], 3)} in {user.moon['house']}\n")
    output += (f"Mercury:   {user.mercury['sign']} {round(user.mercury['position'], 3)} in {user.mercury['house']}\n")
    output += (f"Venus:     {user.venus['sign']} {round(user.venus['position'], 3)} in {user.venus['house']}\n")
    output += (f"Mars:      {user.mars['sign']} {round(user.mars['position'], 3)} in {user.mars['house']}\n")
    output += (f"Jupiter:   {user.jupiter['sign']} {round(user.jupiter['position'], 3)} in {user.jupiter['house']}\n")
    output += (f"Saturn:    {user.saturn['sign']} {round(user.saturn['position'], 3)} in {user.saturn['house']}\n")
    output += (f"Uranus:    {user.uranus['sign']} {round(user.uranus['position'], 3)} in {user.uranus['house']}\n")
    output += (f"Neptune:   {user.neptune['sign']} {round(user.neptune['position'], 3)} in {user.neptune['house']}\n")
    output += (f"Pluto:     {user.pluto['sign']} {round(user.pluto['position'], 3)} in {user.pluto['house']}\n")
    #output += (f"Juno:      {p[10]['sign']} {round(p[10]['pos'], 3)} in {p[10]['house']}\n\n")
    output += ("\nPLACIDUS HOUSES\n")
    output += (f"House Cusp 1:     {user.first_house['sign']}  {round(user.first_house['position'], 3)}\n")
    output += (f"House Cusp 2:     {user.second_house['sign']}  {round(user.second_house['position'], 3)}\n")
    output += (f"House Cusp 3:     {user.third_house['sign']}  {round(user.third_house['position'], 3)}\n")
    output += (f"House Cusp 4:     {user.fourth_house['sign']}  {round(user.fourth_house['position'], 3)}\n")
    output += (f"House Cusp 5:     {user.fifth_house['sign']}  {round(user.fifth_house['position'], 3)}\n")
    output += (f"House Cusp 6:     {user.sixth_house['sign']}  {round(user.sixth_house['position'], 3)}\n")
    output += (f"House Cusp 7:     {user.seventh_house['sign']}  {round(user.seventh_house['position'], 3)}\n")
    output += (f"House Cusp 8:     {user.eighth_house['sign']}  {round(user.eighth_house['position'], 3)}\n")
    output += (f"House Cusp 9:     {user.ninth_house['sign']}  {round(user.ninth_house['position'], 3)}\n")
    output += (f"House Cusp 10:    {user.tenth_house['sign']}  {round(user.tenth_house['position'], 3)}\n")
    output += (f"House Cusp 11:    {user.eleventh_house['sign']}  {round(user.eleventh_house['position'], 3)}\n")
    output += (f"House Cusp 12:    {user.twelfth_house['sign']}  {round(user.twelfth_house['position'], 3)}\n")
    output += ("\n")
    return output


def get_coordinates(address):
        url = f"https://geocode.search.hereapi.com/v1/geocode?q={address}&apiKey={HERE_API_KEY}"
        
        response = requests.get(url)
        json = response.json()
        
        return json["items"][0]["position"]