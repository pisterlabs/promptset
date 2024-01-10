"""
clara_tts_api.py

This module implements text-to-speech (TTS) functionality for various engines, including ReadSpeaker, Google TTS, and ABAIR.

Classes:
- TTSEngine: Base class for TTS engines.
- ReadSpeakerEngine: Derived class for ReadSpeaker TTS engine.
- GoogleTTSEngine: Derived class for Google TTS engine.
- ABAIREngine: Derived class for ABAIR TTS engine.
- IPAReaderEngine: Derived class for ipa-reader phonetic TTS engine.

Functions:
- create_tts_engine(engine_type: str) -> TTSEngine:
Returns an instance of the specified TTS engine type.
- get_tts_engine(language: str) -> Optional[TTSEngine]:
Returns the first available TTS engine that supports the given language.
- get_default_voice(language: str, tts_engine: Optional[TTSEngine]=None) -> Optional[str]:
Returns the default voice for the given language in the specified TTS engine or the first available engine if not specified.
- get_language_id(language: str, tts_engine: Optional[TTSEngine]=None) -> Optional[str]:
Returns the language ID for the given language in the specified TTS engine or the first available engine if not specified.
"""

from .clara_utils import get_config, post_task_update, os_environ_or_none
from .clara_utils import absolute_file_name, absolute_local_file_name, local_file_exists, write_local_txt_file

from openai import OpenAI

import os
import tempfile
import requests
import base64
import gtts
import traceback
import json

config = get_config()

class TTSEngine:
    def create_mp3(self, language_id, voice_id, text, output_file):
        raise NotImplementedError

class ReadSpeakerEngine(TTSEngine):
    def __init__(self, api_key=None, base_url=None):
        self.tts_engine_type = 'readspeaker'
        self.phonetic = False
        self.api_key = api_key or self.load_api_key()
        self.base_url = base_url or config.get('tts', 'readspeaker_base_url')
        self.languages = { 'english':
                            {  'language_id': 'en_uk',
                               'voices': [ 'Alice-DNN' ]
                               },
                            'french':
                            {  'language_id': 'fr_fr',
                               'voices': [ 'Elise-DNN' ]
                               },
                            'italian':
                            {  'language_id': 'it_it',
                               'voices': [ 'Gina-DNN' ]
                               },
                            'german':
                            {  'language_id': 'de_de',
                               'voices': [ 'Max-DNN' ]
                               },
                            'danish':
                            {  'language_id': 'da_dk',
                               'voices': [ 'Lene' ]
                               },
                            'spanish':
                            {  'language_id': 'es_es',
                               'voices': [ 'Pilar-DNN' ]
                               },
                            'icelandic':
                            {  'language_id': 'is_is',
                               'voices': [ 'Female01' ]
                               },
                            'swedish':
                            {  'language_id': 'sv_se',
                               'voices': [ 'Maja-DNN' ]
                               },
                            'farsi':
                            {  'language_id': 'fa_ir',
                               'voices': [ 'Female01' ]
                               },
                            'mandarin':
                            {  'language_id': 'zh_cn',
                               'voices': [ 'Hui' ]
                               },
                            'dutch':
                            {  'language_id': 'nl_nl',
                               'voices': [ 'Ilse-DNN' ]
                               },
                            'japanese':
                            {  'language_id': 'ja_jp',
                               'voices': [ 'Sayaka-DNN' ]
                               },
                            'polish':
                            {  'language_id': 'pl_pl',
                               'voices': [ 'Aneta-DNN' ]
                               },
                            'slovak':
                            {  'language_id': 'sk_sk',
                               'voices': [ 'Jakub' ]
                               }
                          }


    def load_api_key(self):
        try:
            key_path = absolute_local_file_name(config.get('paths', 'readspeaker_license_key'))
            with open(key_path, 'r') as f:
                return f.read().strip()
        except:
            return None
        
    def create_mp3(self, language_id, voice_id, text, output_file, callback=None):
        data = {
            "key": self.api_key,
            "lang": language_id,
            "voice": voice_id,
            "text": text,
            "streaming": 0
        }
        response = requests.post(self.base_url, data, stream=True)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return True
        else:
            return False

class GoogleTTSEngine(TTSEngine):
    def __init__(self):
        self.tts_engine_type = 'google'
        self.phonetic = False
        self.languages =  {'afrikaans': {'language_id': 'af', 'voices': ['default']},
                           'albanian': {'language_id': 'sq', 'voices': ['default']},
                           'arabic': {'language_id': 'ar', 'voices': ['default']},
                           'armenian': {'language_id': 'hy', 'voices': ['default']},
                           'bengali': {'language_id': 'bn', 'voices': ['default']},
                           'bosnian': {'language_id': 'bs', 'voices': ['default']},
                           'catalan': {'language_id': 'ca', 'voices': ['default']},
                           'cantonese': {'language_id': 'zh-CN', 'voices': ['default']},
                           'chinese': {'language_id': 'zh', 'voices': ['default']},
                           'mandarin': {'language_id': 'zh', 'voices': ['default']},
                           'taiwanese': {'language_id': 'zh-TW', 'voices': ['default']},
                           'croatian': {'language_id': 'hr', 'voices': ['default']},
                           'czech': {'language_id': 'cs', 'voices': ['default']},
                           'danish': {'language_id': 'da', 'voices': ['default']},
                           'dutch': {'language_id': 'nl', 'voices': ['default']},
                           'english': {'language_id': 'en', 'voices': ['default']},
                           'esperanto': {'language_id': 'eo', 'voices': ['default']},
                           'estonian': {'language_id': 'et', 'voices': ['default']},
                           'filipino': {'language_id': 'tl', 'voices': ['default']},
                           'finnish': {'language_id': 'fi', 'voices': ['default']},
                           'french': {'language_id': 'fr', 'voices': ['default']},
                           'german': {'language_id': 'de', 'voices': ['default']},
                           'greek': {'language_id': 'el', 'voices': ['default']},
                           'gujarati': {'language_id': 'gu', 'voices': ['default']},
                           # In fact, not yet available
                           #'hebrew': {'language_id': 'he-IL', 'voices': ['default']},
                           'hindi': {'language_id': 'hi', 'voices': ['default']},
                           'hungarian': {'language_id': 'hu', 'voices': ['default']},
                           'icelandic': {'language_id': 'is', 'voices': ['default']},
                           'indonesian': {'language_id': 'id', 'voices': ['default']},
                           'italian': {'language_id': 'it', 'voices': ['default']},
                           'japanese': {'language_id': 'ja', 'voices': ['default']},
                           'javanese': {'language_id': 'jw', 'voices': ['default']},
                           'kannada': {'language_id': 'kn', 'voices': ['default']},
                           'khmer': {'language_id': 'km', 'voices': ['default']},
                           'korean': {'language_id': 'ko', 'voices': ['default']},
                           'latin': {'language_id': 'la', 'voices': ['default']},
                           'latvian': {'language_id': 'lv', 'voices': ['default']},
                           'macedonian': {'language_id': 'mk', 'voices': ['default']},
                           'malayalam': {'language_id': 'ml', 'voices': ['default']},
                           'marathi': {'language_id': 'mr', 'voices': ['default']},
                           'burmese': {'language_id': 'my', 'voices': ['default']},
                           'nepali': {'language_id': 'ne', 'voices': ['default']},
                           'norwegian': {'language_id': 'no', 'voices': ['default']},
                           'polish': {'language_id': 'pl', 'voices': ['default']},
                           'portuguese': {'language_id': 'pt', 'voices': ['default']},
                           'romanian': {'language_id': 'ro', 'voices': ['default']},
                           'russian': {'language_id': 'ru', 'voices': ['default']},
                           'serbian': {'language_id': 'sr', 'voices': ['default']},
                           'sinhala': {'language_id': 'si', 'voices': ['default']},
                           'slovak': {'language_id': 'sk', 'voices': ['default']},
                           'spanish': {'language_id': 'es', 'voices': ['default']},
                           'sundanese': {'language_id': 'su', 'voices': ['default']},
                           'swahili': {'language_id': 'sw', 'voices': ['default']},
                           'swedish': {'language_id': 'sv', 'voices': ['default']},
                           'tamil': {'language_id': 'ta', 'voices': ['default']},
                           'telugu': {'language_id': 'te', 'voices': ['default']},
                           'thai': {'language_id': 'th', 'voices': ['default']},
                           'turkish': {'language_id': 'tr', 'voices': ['default']},
                           'ukrainian': {'language_id': 'uk', 'voices': ['default']},
                           'urdu': {'language_id': 'ur', 'voices': ['default']},
                           'vietnamese': {'language_id': 'vi', 'voices': ['default']},
                           'welsh': {'language_id': 'cy', 'voices': ['default']}
                        }

    def create_mp3(self, language_id, voice_id, text, output_file, callback=None):
        try:
            found_google_creds = self._load_google_application_creds(callback=callback)

            if found_google_creds:
                tts = gtts.gTTS(text, lang=language_id)
                tts.save(output_file)
                return True
##        except gtts.GTTSError as e:
##            post_task_update(callback, f"*** Warning: gTTS error while creating Google TTS mp3 for '{text}': {str(e)}")
##            return False
        except requests.exceptions.RequestException as e:
            post_task_update(callback, f"*** Warning: Network error while creating Google TTS mp3 for '{text}': {str(e)}")
            return False
        except IOError as e:
            post_task_update(callback, f"*** Warning: IOError while saving Google TTS mp3 for '{text}': {str(e)}")
            return False
        except Exception as e:
            post_task_update(callback, f"*** Warning: unable to create Google TTS mp3 for '{text}': {str(e)}")
            return False

    def _load_google_application_creds(self, callback=None):
        creds_file = os_environ_or_none('GOOGLE_APPLICATION_CREDENTIALS')
        creds_string = os_environ_or_none('GOOGLE_CREDENTIALS_JSON')
        
        if creds_file and local_file_exists(creds_file):
            return True
        elif creds_string:
            creds_file = '/tmp/google_credentials_from_env.json'
            write_local_txt_file(creds_string, creds_file)                
            # Set the environment variable so gTTS can pick it up
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file
            return True 
        else:
            post_task_update(callback, f"*** Warning: unable to find Google credentials in GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CREDENTIALS_JSON")
            return False

class OpenAITTSEngine(TTSEngine):
    def __init__(self):
        self.tts_engine_type = 'openai'
        self.open_ai_key = os.environ["OPENAI_API_KEY"]
        self.phonetic = False
        self._openai_supported_languages = ['afrikaans', 'arabic', 'armenian', 'azerbaijani', 'belarusian',
                                            'bosnian', 'bulgarian', 'catalan', 'chinese', 'croatian', 'czech',
                                            'danish', 'dutch', 'english', 'estonian', 'finnish', 'french',
                                            'galician', 'german', 'greek', 'hebrew', 'hindi', 'hungarian',
                                            'icelandic', 'indonesian', 'italian', 'japanese', 'kannada', 'kazakh',
                                            'korean', 'latvian', 'lithuanian', 'macedonian', 'malay', 'marathi',
                                            'māori', 'nepali', 'norwegian', 'persian', 'polish', 'portuguese',
                                            'romanian', 'russian', 'serbian', 'slovak', 'slovenian', 'spanish',
                                            'swahili', 'swedish', 'tagalog', 'tamil', 'thai', 'turkish', 'ukrainian',
                                            'urdu', 'vietnamese', 'welsh']
        self._openai_supported_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
                                            
        self.languages =  { language: {'language_id': language, 'voices': self._openai_supported_voices }
                            for language in self._openai_supported_languages }
                           
    def create_mp3(self, language_id, voice_id, text, output_file, callback=None):
        try: 
            client = OpenAI(api_key=self.open_ai_key)
            speech_file_path = absolute_local_file_name(output_file)
            response = client.audio.speech.create(
                              model="tts-1",
                              voice=voice_id,
                              input=text
                              )
            response.stream_to_file(speech_file_path)
            return True
        
        except requests.exceptions.RequestException as e:
            post_task_update(callback, f"*** Warning: Network error while creating OpenAI TTS mp3 for '{text}': '{str(e)}'\n{traceback.format_exc()}")
            return False
        except IOError as e:
            post_task_update(callback, f"*** Warning: IOError while saving OpenAI TTS mp3 for '{text}': '{str(e)}'\n{traceback.format_exc()}")
            return False
        except Exception as e:
            post_task_update(callback, f"*** Warning: unable to create OpenAI TTS mp3 for '{text}': '{str(e)}'\n{traceback.format_exc()}")
            return False

class ABAIREngine(TTSEngine):
    def __init__(self, base_url=None):
        self.tts_engine_type = 'abair'
        self.phonetic = False
        self.base_url = base_url or config.get('tts', 'abair_base_url')
        self.languages = { 'irish':
                            {  'language_id': 'ga-IE',
                               'voices': [
                                   'ga_UL_anb_nnmnkwii',
                                   'ga_MU_nnc_nnmnkwii',
                                   'ga_MU_cmg_nnmnkwii'                           
                               ]
                            }
                          }

    def create_mp3(self, language_id, voice_id, text, output_file, callback=None):
        data = {
            "synthinput": {
                "text": text
            },
            "voiceparams": {
                "languageCode": language_id,
                "name": voice_id
            },
            "audioconfig": {
                "audioEncoding": "MP3"
            }
        }
        response = requests.post(self.base_url, json=data)
        if response.status_code == 200:
            encoded_audio = response.json()["audioContent"]
            decoded_audio = base64.b64decode(encoded_audio)
            with open(output_file, 'wb') as f:
                f.write(decoded_audio)
            return True
        else:
            return False

## Use ipa-reader.xyz to create an mp3 for a piece of IPA text.
##
## Code slightly adapted from a solution found by Claudia

class IPAReaderEngine(TTSEngine):
    def __init__(self):
        self.tts_engine_type = 'ipa_reader'
        self.phonetic = True
        self.execute_url = "https://iawll6of90.execute-api.us-east-1.amazonaws.com/production"
        
        self.languages = { 'american': { 'language_id': 'american',
                                         'voices': [ 'Salli',
                                                     'Ivy',
                                                     'Joanna',
                                                     'Joey',
                                                     'Justin',
                                                     'Kendra',
                                                     'Kimberley'
                                                     ]
                                         },
                           'english': { 'language_id': 'american',
                                         'voices': [ 'Emma',
                                                     'Brian',
                                                     'Amy'
                                                     ]
                                         },
                           'australian': { 'language_id': 'australian',
                                           'voices': [ 'Nicole',
                                                       'Russell'
                                                       ]
                                           },
                           'french': { 'language_id': 'french',
                                       'voices': [ 'Celine',
                                                   'Mathieu'
                                                   ]
                                       },
                           'icelandic': { 'language_id': 'icelandic',
                                          'voices': [ 'Karl',
                                                      'Dora'
                                                      ]
                                          },
                           'romanian': { 'language_id': 'romanian',
                                         'voices': [ 'Carmen'
                                                      ]
                                         },
                           'dutch': { 'language_id': 'dutch',
                                      'voices': [ 'Lotte',
                                                  'Ruben'
                                                  ]
                                      },
                           'portuguese': { 'language_id': 'portuguese',
                                           'voices': [ 'Cristiano',
                                                       'Ines'
                                                       ]
                                      },
                           'german': { 'language_id': 'german',
                                       'voices': [ 'Marlene'
                                                   ]
                                      },
                           'italian': { 'language_id': 'italian',
                                        'voices': [ 'Carla',
                                                    'Giorgio'
                                                    ]
                                      },
                           'japanese': { 'language_id': 'japanese',
                                         'voices': [ 'Mizuki'
                                                     ]
                                      },
                           'norwegian': { 'language_id': 'norwegian',
                                          'voices': [ 'Liv'
                                                      ]
                                      },
                           'polish': { 'language_id': 'polish',
                                       'voices': [ 'Maja',
                                                   'Jan',
                                                   'Ewa'
                                                   ]
                                      },
                           'russian': { 'language_id': 'russian',
                                        'voices': [ 'Maxim',
                                                    'Tatyana'
                                                    ]
                                      },
                           'spanish': { 'language_id': 'spanish',
                                        'voices': [ 'Conchita'
                                                    ]
                                      },
                           'swedish': { 'language_id': 'swedish',
                                        'voices': [ 'Astrid'
                                                    ]
                                      },
                           'turkish': { 'language_id': 'turkish',
                                        'voices': [ 'Filiz'
                                                    ]
                                      },
                           'welsh': { 'language_id': 'welsh',
                                      'voices': [ 'Gwyneth'
                                                    ]
                                      },

                           }

    def create_mp3(self, language_id, voice_id, text, output_file, callback=None):
        try:
            payload = json.dumps({
              "text": f"/{text}/",  
              "voice": voice_id
            })
            
            headers = {
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
              'Accept': '*/*',
              'Accept-Language': 'en-US,en;q=0.5',
              'Accept-Encoding': 'gzip, deflate, br',
              'Content-Type': 'application/json',
              'Origin': 'http://ipa-reader.xyz',
              'Connection': 'keep-alive',
              'Referer': 'http://ipa-reader.xyz/',
              'Sec-Fetch-Dest': 'empty',
              'Sec-Fetch-Mode': 'cors',
              'Sec-Fetch-Site': 'cross-site',
              'TE': 'trailers'
            }

            response = requests.request("POST", self.execute_url, headers=headers, data=payload)

            binary_data = response._content.decode('unicode_escape')

            # Decode the base64-encoded binary data
            decoded_data = base64.b64decode(binary_data)

            abs_output_file = absolute_local_file_name(output_file)

            with open(abs_output_file, "wb") as audio_file:
                audio_file.write(decoded_data)

            return True
                
        except requests.exceptions.RequestException as e:
            post_task_update(callback, f"*** Warning: Network error while creating ipa-reader mp3 for '{text}': '{str(e)}'\n{traceback.format_exc()}")
            return False
        except IOError as e:
            post_task_update(callback, f"*** Warning: IOError while saving ipa-reader mp3 for '{text}': '{str(e)}'\n{traceback.format_exc()}")
            return False
        except Exception as e:
            post_task_update(callback, f"*** Warning: unable to create ipa-reader mp3 for '{text}': '{str(e)}'\n{traceback.format_exc()}")
            return False


TTS_ENGINES = [ABAIREngine(), GoogleTTSEngine(), OpenAITTSEngine(), ReadSpeakerEngine(), IPAReaderEngine()]

def create_tts_engine(engine_type):
    if engine_type == 'readspeaker':
        return ReadSpeakerEngine()
    elif engine_type == 'google':
        return GoogleTTSEngine()
    elif engine_type == 'abair':
        return ABAIREngine()
    elif engine_type == 'openai':
        return OpenAITTSEngine()
    elif engine_type == 'ipa_reader':
        return IPAReaderEngine()
    else:
        raise ValueError(f"Unknown TTS engine type: {engine_type}")
    
def get_tts_engine(language, phonetic=False, callback=None):
    for tts_engine in TTS_ENGINES:
        if language in tts_engine.languages and tts_engine.phonetic == phonetic:
            post_task_update(callback, f"--- clara_tts_api found TTS engine of type '{tts_engine.tts_engine_type}'")
            return tts_engine
    return None

def get_tts_engine_types():
    return [ tts_engine.tts_engine_type for tts_engine in TTS_ENGINES ]

def get_default_voice(language, tts_engine=None):
    tts_engine = tts_engine or get_tts_engine(language)
    if tts_engine and language in tts_engine.languages:
        return tts_engine.languages[language]['voices'][0]
    return None

def get_language_id(language, tts_engine=None):
    tts_engine = tts_engine or get_tts_engine(language)
    if tts_engine and language in tts_engine.languages:
        return tts_engine.languages[language]['language_id']
    return None
