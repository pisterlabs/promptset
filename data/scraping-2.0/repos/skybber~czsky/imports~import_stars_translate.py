from datetime import datetime
import time
import openai

from flask import current_app

from os import listdir, getenv

from sqlalchemy.exc import IntegrityError

from app import db

from app.models.star import UserStarDescription
from app.models.user import User


def import_stars_translate():
    gpt_prompt = '''Přelož následující astronomický text do angličtiny. České zkratky světových stran (S,Z,J,E) a přelož do českých zkratek (N,W,S,E). 
Zkratky světových stran nikdy nepřekládej do jejich slovních ekvivalentů. Kombinace zkratek světových stran překládej vždy na kombinace zkratek v angličtině, například 
ZSJ přelož na WNW, VJV na ESE, JV na SE, ZJZ na WSW, VSV na ENE, SZ na NW, JJZ na SSW, SSV na NNE, SV na NE. Nepřekládej IAU kód souhvězdí. Boční vidění překládej jako "averted vision". 

'''
    openai.api_key = getenv("OPENAI_API_KEY")

    user_editor_cs = User.query.filter_by(user_name=current_app.config.get('EDITOR_USER_NAME_CS')).first()

    descrs = UserStarDescription.query.filter_by(user_id=user_editor_cs.id, lang_code='cs')

    try:
        for descr_cs in descrs:
            descr_en = UserStarDescription.query.filter_by(user_id=user_editor_cs.id, common_name=descr_cs.common_name, lang_code='en').first()
            if descr_en is None:
                descr_en = UserStarDescription(
                    constellation_id=descr_cs.constellation_id,
                    common_name=descr_cs.common_name,
                    user_id=user_editor_cs.id,
                    lang_code='en',
                    star_id=descr_cs.star_id,
                    create_by=user_editor_cs.id,
                    update_by=user_editor_cs.id,
                    create_date=datetime.now(),
                    update_date=datetime.now(),
                    double_star_id=descr_cs.double_star_id
                )
                print('Translating: {}'.format(descr_cs.text))
                while True:
                    try:
                        messages = [{"role": "user", "content": gpt_prompt + descr_cs.text}]

                        completion = openai.ChatCompletion.create(
                            model='gpt-3.5-turbo-1106',
                            messages=messages,
                            temperature=0.0
                        )

                        descr_en.text = completion.choices[0].message.content
                        print('Translated: {}'.format(descr_en.text))
                        break
                    except:
                        print('Sleeeping...')
                        time.sleep(10)
                db.session.add(descr_en)
                db.session.flush()
                db.session.commit()
    except IntegrityError as err:
        print('\nIntegrity error {}'.format(err))
        db.session.rollback()
