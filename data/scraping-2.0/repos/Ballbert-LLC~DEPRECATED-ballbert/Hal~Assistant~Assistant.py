import os
import sqlite3
import openai

from Config import Config
from ..Logging import display_error, handle_error

from ..Classes import Response, NoTTSException, NoVoiceException
from ..Logging import log_line
from ..Memory import Weaviate
from ..MessageHandler import MessageHandler
from ..TTS import TTS
from ..Voice import Voice
from .SkillMangager import SkillMangager

repos_path = f"{os.path.abspath(os.getcwd())}/Skills"

config = Config()

openai.api_key = config["OPENAI_API_KEY"]


class Assistant:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # pinecone memory
        pm = None
        skill_manager = None
        try:
            pm = Weaviate()
            tts = self.setup_tts()
            voice = self.setup_voice()
            skill_manager = SkillMangager()
        except Exception as e:
            log_line("Err", e)
            display_error()
            handle_error()
        self.messages = []
        self.pm = pm
        self.tts = tts
        self.voice = voice
        self.installed_skills = dict()
        self.action_dict = dict()
        self.skill_manager = skill_manager
        self.voice = voice
        self.tts = tts
        self.speak_mode = False
        self.current_callback = None

    async def sentance_gen(self, res):
        try:
            buffer = ""
            index = 0
            async for content in self._text_gpt_response(res):
                for char in content:
                    if char in [".", "!", "?"]:
                        buffer += char
                        yield (buffer, index)
                        index += 1
                        buffer = ""
                    else:
                        buffer += char
        except Exception as e:
            log_line("Err", e)
            yield "Sorry i made a mistake can you say that again."

    def setup_tts(self):
        try:
            tts = TTS()
            return tts
        except Exception as e:
            log_line("Err", e)
            return None

    def setup_voice(self):
        try:
            voice = Voice()
            return voice
        except Exception as e:
            log_line("Err", e)
            return None

    def voice_to_voice_chat(self):
        async def callback(res, err):
            gen = self.sentance_gen(res)

            try:
                if self.tts == None:
                    raise NoTTSException("No tts")

                await self.tts.speak_gen(gen)
            except NoTTSException as e:
                log_line("Err", e)
                self.tts = self.setup_tts()
                self.tts.backup_speaking()
            except Exception as e:
                log_line("Err", e)
                self.tts = self.setup_tts()

        try:
            print(self.voice)
            if self.voice == None:
                print("no voice", self.voice)
                raise NoVoiceException("No Voice")
            self.voice.start(callback)
        except NoVoiceException as e:
            log_line("Err", e)
            self.voice = self.setup_voice()
            self.voice_to_voice_chat()
        except Exception as e:
            log_line("Err", e)
            display_error()
            handle_error()

    async def _text_gpt_response(self, to_gpt):
        message_handler = MessageHandler(self, to_gpt)

        async for chunk in message_handler.handle_message():
            yield chunk

    def call_function(self, function_id, args=[], kwargs={}):
        try:
            return self.action_dict[function_id]["function"](*args, **kwargs)
        except Exception as e:
            log_line("Err", e)
            return Response(succeeded=False)


# Module-level variable to store the shared instance
assistant = None


# Initialization function to create the instance
def initialize_assistant():
    def setup_assistant():
        con = sqlite3.connect("skills.db")

        cur = con.cursor()
        try:
            # Execute a SELECT query on the installedSkills table
            cur.execute("SELECT * FROM installedSkills")
            installed_skills_data = cur.fetchall()
        except:
            installed_skills_data = []

        for item in installed_skills_data:
            assistant.skill_manager.add_skill(assistant, item[0])

        con.commit()

        con.close()

    global assistant
    if assistant is None:
        assistant = Assistant()

        try:
            setup_assistant()
        except Exception as e:
            log_line("Err", e)
            display_error()
            handle_error()

        return assistant

    else:
        return assistant
