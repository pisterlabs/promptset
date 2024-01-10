from datetime import datetime
import time
import requests
from sqlalchemy.exc import IntegrityError
from astroquery.simbad import Simbad
import openai
import tiktoken

from flask import (
    current_app,
)

from app import db

from app.models.deepskyobject import DeepskyObject, UserDsoDescription, UserDsoApertureDescription
from app.models.user import User

WIKIPEDIA_REF = '{} Wikipedia'

class TranslHolder:
    def __init__(self, gpt_prompt):
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-1106')
        self.gpt_prompt = gpt_prompt
        self.dso_names = []
        self.descrs = []
        self.texts = []
        self.string = ''

    def add_descr(self, dso_name, descr, t):
        if self._should_flush(t):
            self.flush()
        self.dso_names.append(dso_name)
        self.string += self._format_part(t)
        self.descrs.append(descr)
        self.texts.append(t)

    def _format_part(self, t):
        return '__{}__\n{}\n'.format(len(self.texts), t)

    def _should_flush(self, t):
        if len(self.texts) == 0:
            return False
        new_string = self.string + self._format_part(t)
        num_tokens = len(self.encoding.encode(self.gpt_prompt + new_string))
        return num_tokens > 1300

    def _translate_it(self, text):
        while True:
            try:
                messages = [{"role": "user", "content": self.gpt_prompt + text}]

                completion = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-1106',
                    messages=messages,
                    temperature=0.0
                )

                translated = completion.choices[0].message.content
                return translated
            except:
                print('Sleeeping...')
                time.sleep(10)

    def flush(self):
        if len(self.string) == 0:
            return

        print('Translating {} '.format(self.dso_names))

        required_tag = '__{}__'.format(len(self.texts))

        self.string += required_tag

        transl = self._translate_it(self.string).strip()

        if not required_tag in transl:
            print('Bulk translation failed!')
            for i, t in enumerate(self.texts):
                d = self.descrs[i]
                tr = self._translate_it(t).strip()
                print('{}\n{}\n'.format(t, tr))
                d.text = tr
                db.session.add(d)
                db.session.flush()
        else:
            if not transl.startswith('__0__'):
                transl = '__0__\n' + transl

            for i, d in enumerate(self.descrs):
                tag1 = '__{}__'.format(i)
                tag2 = '__{}__'.format(i+1)
                i1 = transl.find(tag1)
                i2 = transl.find(tag2)
                i1_orig = self.string.find(tag1)
                i2_orig = self.string.find(tag2)

                if i1 == -1 or i1_orig == -1:
                    print('Tag __x__ not found {}'.format(tag1))
                    exit()

                if i2 == -1 or i2_orig == -1:
                    print('Tag __x__ not found {}'.format(tag2))
                    exit()

                t = transl[i1+len(tag1):i2].strip()

                t_orig = self.string[i1_orig+len(tag1):i2_orig].strip()

                print('{}\n{}\n'.format(t_orig, t))

                d.text = t
                db.session.add(d)
                db.session.flush()

        db.session.commit()

        self.dso_names = []
        self.descrs = []
        self.texts = []
        self.string = ''

def import_wikipedia_ngc(do_update=False):

    simbad = Simbad()
    simbad.add_votable_fields('otype')

    user_wikipedia = User.query.filter_by(user_name=current_app.config.get('EDITOR_USER_NAME_WIKIPEDIA')).first()
    user_editor_cs = User.query.filter_by(user_name=current_app.config.get('EDITOR_USER_NAME_CS')).first()

    if not user_wikipedia:
        print('User editor.wikipedia not found!')
        return

    try:
        for i in range(1, 7841):
            dso_name = 'NGC{}'.format(i)
            czsky_name = dso_name

            dso = DeepskyObject.query.filter_by(name=dso_name).first()

            if not dso:
                dso = DeepskyObject.query.filter_by(name=dso_name + 'A').first()
                if dso:
                    czsky_name = czsky_name + 'A'

            if not dso:
                dso = DeepskyObject.query.filter_by(name=dso_name + '_1').first()
                if dso:
                    czsky_name = czsky_name + '_1'

            if dso and dso.master_id:
                dso = DeepskyObject.query.filter_by(id=dso.master_id).first()

            try:
                resp = requests.get('https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles=NGC%20{}'.format(i))
            except ConnectionError:
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()

            found = False
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                for page_id, page_data in pages.items():
                    if 'extract' in page_data:
                        extract = page_data['extract'].strip()
                        if extract:
                            parts = extract.split('\n\n\n')
                            if len(parts) >= 1:
                                dso_descr = parts[0].strip()

                                found = True

                                if not dso:
                                    print('{} not exist. {}'.format(dso_name, dso_descr), flush=True)
                                    break

                                mag8_descr = UserDsoDescription.query.filter_by(dso_id=dso.id, user_id=user_editor_cs.id).first()

                                udd = UserDsoDescription.query.filter_by(dso_id=dso.id, user_id=user_wikipedia.id, lang_code='en').first()

                                if udd and not do_update:
                                    continue

                                if udd:
                                    print('Updating data {}'.format(dso_name))
                                    udd.text = dso_descr
                                    udd.update_by = user_wikipedia.id
                                    udd.update_date = datetime.now()
                                else:
                                    print('Inserting data {}'.format(dso_name))
                                    udd = UserDsoDescription(
                                        dso_id=dso.id,
                                        user_id=user_wikipedia.id,
                                        rating=mag8_descr.rating if mag8_descr else 0,
                                        lang_code='en',
                                        cons_order=mag8_descr.cons_order if mag8_descr else 100000,
                                        text=dso_descr,
                                        references=None,
                                        common_name=dso.common_name,
                                        create_by=user_wikipedia.id,
                                        update_by=user_wikipedia.id,
                                        create_date=datetime.now(),
                                        update_date=datetime.now(),

                                    )
                                db.session.add(udd)
                                db.session.flush()
                                db.session.commit()
            if dso and not found:
                print('{} not found on wiki. czsky_name={}'.format(dso_name, czsky_name), flush=True)
        db.session.flush()
        db.session.commit()
    except KeyError as err:
        print('\nKey error: {}'.format(err))
        db.session.rollback()
    except IntegrityError as err:
        print('\nIntegrity error {}'.format(err))
        db.session.rollback()

def translate_wikipedia_ngc(lang_code, ref_prefix, gpt_prompt):

    user_editor_cs = User.query.filter_by(user_name=current_app.config.get('EDITOR_USER_NAME_CS')).first()
    user_wikipedia = User.query.filter_by(user_name=current_app.config.get('EDITOR_USER_NAME_WIKIPEDIA')).first()

    if not user_wikipedia:
        print('User editor.wikipedia not found!')
        return

    try:
        tholder = TranslHolder(gpt_prompt)
        user_descr = UserDsoDescription.query.filter_by(user_id=user_wikipedia.id, lang_code='en').all()

        for udd_en in user_descr:
            mag8_descr = UserDsoDescription.query.filter_by(dso_id=udd_en.dso_id, user_id=user_editor_cs.id).first()
            udd = UserDsoDescription.query.filter_by(dso_id=udd_en.dso_id, user_id=user_wikipedia.id, lang_code=lang_code).first()
            if not udd:
                dso = DeepskyObject.query.filter_by(id=udd_en.dso_id).first()

                if dso and dso.master_id:
                    dso = DeepskyObject.query.filter_by(id=dso.master_id).first()
                udd = UserDsoDescription(
                    dso_id=udd_en.dso_id,
                    user_id=user_wikipedia.id,
                    rating=mag8_descr.rating if mag8_descr else 0,
                    lang_code=lang_code,
                    cons_order=mag8_descr.cons_order if mag8_descr else 100000,
                    text=None,
                    references=WIKIPEDIA_REF.format(ref_prefix),
                    common_name=dso.common_name,
                    create_by=user_wikipedia.id,
                    update_by=user_wikipedia.id,
                    create_date=datetime.now(),
                    update_date=datetime.now(),

                )
                tholder.add_descr(dso.name, udd, udd_en.text)

        tholder.flush()
        db.session.flush()
        db.session.commit()
    except KeyError as err:
        print('\nKey error: {}'.format(err))
        db.session.rollback()
    except IntegrityError as err:
        print('\nIntegrity error {}'.format(err))
        db.session.rollback()
