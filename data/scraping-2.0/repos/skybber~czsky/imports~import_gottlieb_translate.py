from datetime import datetime
import re
import time
import openai
import tiktoken

from flask import current_app

from os import listdir, getenv
from os.path import isfile, join

from sqlalchemy.exc import IntegrityError

from app import db

from app.models.deepskyobject import DeepskyObject, UserDsoDescription, UserDsoApertureDescription
from app.models.user import User


GOTTLIEB_REF = '{} [Steve Gottlieb](https://www.astronomy-mall.com/Adventures.In.Deep.Space/steve.ngc.htm)'

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


def _found_dso(dso_name, dso_descr, dso_apert_descr, user_editor_en, user_editor_cs, lang_code, ref_prefix, tholder):

    dso = DeepskyObject.query.filter_by(name=dso_name).first()

    if dso and dso.master_id:
        dso = DeepskyObject.query.filter_by(id=dso.master_id).first()

    if dso:
        mag8_descr = UserDsoDescription.query.filter_by(dso_id=dso.id, user_id=user_editor_cs.id).first()

        # update constellation ID since it is missing in vic catalogue

        # print('Importing {}'.format(dso_name))

        udd = UserDsoDescription.query.filter_by(dso_id=dso.id, user_id=user_editor_en.id, lang_code=lang_code).first()
        if False: #not udd:
            udd = UserDsoDescription(
                dso_id=dso.id,
                user_id=user_editor_en.id,
                rating=mag8_descr.rating if mag8_descr else 0,
                lang_code=lang_code,
                cons_order=mag8_descr.cons_order if mag8_descr else 100000,
                text=None,
                references=GOTTLIEB_REF.format(ref_prefix),
                common_name=dso.common_name,
                create_by=user_editor_en.id,
                update_by=user_editor_en.id,
                create_date=datetime.now(),
                update_date=datetime.now(),

            )
            tholder.add_descr(dso_name, udd, apert_descr)


        for apert, apert_descr in dso_apert_descr.items():
            if apert == 'Naked-eye':
                apert_class = 'Naked-eye'
            else:
                apertf = float(apert)
                if apertf < 8:
                    apert_class = '100/150'
                elif apertf < 12:
                    apert_class = '200/250'
                elif apertf < 16:
                    apert_class = '300/350'
                elif apertf < 24:
                    apert_class = '400/500'
                elif apertf < 36:
                    apert_class = '600/800'
                else:
                    apert_class = '900/1200'

            uad = UserDsoApertureDescription.query.filter_by(dso_id=dso.id, user_id=user_editor_en.id, aperture_class=apert_class, lang_code=lang_code).first()

            if not uad:
                uad = UserDsoApertureDescription(
                    dso_id=dso.id,
                    user_id=user_editor_en.id,
                    lang_code=lang_code,
                    aperture_class=apert_class,
                    text=None,
                    is_public=True,
                    create_by=user_editor_en.id,
                    update_by=user_editor_en.id,
                    create_date=datetime.now(),
                    update_date=datetime.now(),
                )

                tholder.add_descr(dso_name, uad, apert_descr)

def import_translated_gottlieb(gottlieb_dir, lang_code, ref_prefix):

    gpt_prompt = '''Přelož následující text astronomického pozorování do češtiny. Anglické zkratky světových stran (N,W,S,E) a přelož do českých zkratek (S,Z,J,V). 
Zkratky světových stran nikdy nepřekládej do jejich slovních ekvivalentů. Kombinace zkratek světových stran překládej vždy na kombinace zkratek v češtině, například 
WNW přelož na ZSZ, ESE na VJV, SE na JV, WSW na ZJZ, ENE na VSV, NW na SZ, SSW na JJZ, NNE na SSV, NE na SV. Kombinace směrů typu NNW-SSE překládej jako kompinaci 
typu SSZ-JJV, SW-NE jako JZ-SV, WSW-ENE jako ZJZ-VSV. Nepřekládej IAU kód souhvězdí. Používej rod ženský jako výchozí rod - tedy místo "malý" piš "malá". 
Překládaná text začína sekvencí __0__. Nikdy neodstraňuj značky typu __0__. 
"Averted vision" překládej na "boční pohled". "tidal tail" na "slapový chvost", "edge on" na "viděna z boku", "bar" na "přička","seeing" na "seeing", "halo" na "halo":

'''

    openai.api_key = getenv("OPENAI_API_KEY")

    user_editor_en = User.query.filter_by(user_name=current_app.config.get('EDITOR_USER_NAME_EN')).first()
    user_editor_cs = User.query.filter_by(user_name=current_app.config.get('EDITOR_USER_NAME_CS')).first()

    gottlieb_files = [f for f in listdir(gottlieb_dir) if isfile(join(gottlieb_dir, f))]

    tholder = TranslHolder(gpt_prompt)

    try:
        print('{}'.format(gottlieb_files))
        for filename in gottlieb_files:
            file = open(join(gottlieb_dir, filename), 'r')
            lines = file.readlines()

            i = 0

            search_obj_id = True

            dso_name = None
            dso_apert_descr = {}
            dso_descr = ''
            last_apert = None
            ignored_text = True

            while i < len(lines):
                line = lines[i]
                i += 1

                if len(line.strip()) == 0:
                    continue

                if line.strip() == '******************************':
                    if dso_name:
                        _found_dso(dso_name, dso_descr, dso_apert_descr, user_editor_en, user_editor_cs, lang_code, ref_prefix, tholder)
                    search_obj_id = True
                    dso_name = None
                    dso_descr = ''
                    dso_apert_descr = {}
                    last_apert = None
                    ignored_text = True
                    continue

                if search_obj_id:
                    if filename.startswith('UGC'):
                        pobj_id = re.match(r'(UGC)\s*(\d+).*', line)
                    else:
                        pobj_id = re.match(r'(NGC|IC)\s*(\d+).*', line)
                    if pobj_id:
                        search_obj_id = False
                        dso_name = pobj_id.group(1) + pobj_id.group(2)
                    continue

                papertd = re.match(r'((?:\d+(?:\.\d+)?")|(?:Naked-eye))\:?(.*)', line)
                if papertd:
                    if last_apert:
                        if dso_descr:
                            dso_apert_descr[last_apert] = dso_apert_descr[last_apert] + '\n\n' + dso_descr
                            dso_descr = ''
                    last_apert = papertd.group(1)
                    apert_prefix = ''
                    if last_apert != 'Naked-eye':
                        last_apert = last_apert[:-1]
                        apert_prefix = papertd.group(1) + ' '
                    if last_apert in dso_apert_descr:
                        dso_apert_descr[last_apert] = dso_apert_descr[last_apert] + '\n\n' + apert_prefix + papertd.group(2)
                    else:
                        dso_apert_descr[last_apert] = apert_prefix + papertd.group(2)
                    ignored_text = False
                    continue

                if ignored_text:
                    continue

                if len(dso_descr) > 0:
                    dso_descr += '\n'

                dso_descr += line

        if dso_name:
            _found_dso(dso_name, dso_descr, dso_apert_descr, user_editor_en, user_editor_cs, lang_code, ref_prefix, tholder)

        tholder.flush()
    except KeyError as err:
        print('\nKey error: {}'.format(err))
        db.session.rollback()
    except IntegrityError as err:
        print('\nIntegrity error {}'.format(err))
        db.session.rollback()
