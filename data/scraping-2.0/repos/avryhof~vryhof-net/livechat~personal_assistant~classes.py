import glob
import os
from importlib import import_module

import openai
from django.conf import settings

from livechat.devices.location.device import GeoIP
from livechat.personal_assistant.base_class import BaseClass
from utilities.utility_functions import is_empty

openai.api_key = getattr(settings, "OPENAI_API_KEY")


class Bot(BaseClass):
    heard = None
    responded = None

    chat_session = None
    geo = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        load_skills = kwargs.get("load_skills", True)

        self.log("Initializing")

        self.chat_session = kwargs.get("chat_session")

        if not is_empty(self.chat_session.latitude) and not is_empty(self.chat_session.longitude):
            self.log("Loading location")
            self.geo = self.chat_session.geo
            self.geo.get_location()

        if not is_empty(kwargs.get("latitude")) and not is_empty(kwargs.get("longitude")):
            self.log("Loading location")
            self.chat_session.latitude = kwargs.get("latitude")
            self.chat_session.longitude = kwargs.get("longitude")
            self.chat_session.save()
            self.geo = self.chat_session.geo
            self.geo.get_location()

        if is_empty(self.geo):
            self.log("Updating Location")
            self.geo = GeoIP()
            self.geo.get_location(kwargs.get("client_ip"))
            self.chat_session.latitude = self.geo.latitude
            self.chat_session.longitude = self.geo.longitude
            self.chat_session.save()

        if not is_empty(self.geo):
            setattr(settings, "LOCATION", self.geo)

        if load_skills:
            self.log("Loading Skills")
            self._init_skills()

        self.log("Initialized Bot.")

    def _init_skills(self):
        skills_path_name = "skills"
        skills_module_name = "skills"
        skills_path = os.path.join(settings.BASE_DIR, "livechat", skills_path_name)
        skills_directory = os.path.join(settings.BASE_DIR, skills_path)

        skills_directories = [
            directory
            for directory in glob.glob(os.path.join(skills_directory, "*"))
            if os.path.isdir(directory) and "__pycache__" not in directory
        ]

        for skill_path in skills_directories:
            skill_module_name = os.path.join(skill_path, f"{skills_module_name}.py")

            if os.path.exists(skill_module_name):
                skill_module_path = "{}.{}".format(
                    skill_path.replace(str(settings.BASE_DIR), "").replace(os.path.sep, "."), skills_module_name
                )[1::]
                skill_module = import_module(skill_module_path)

                for skill_class_name in dir(skill_module):
                    if skill_class_name != "AssistantSkill":
                        skill_class = getattr(skill_module, skill_class_name)

                        if hasattr(skill_class, "assistant_skill"):
                            if hasattr(skill_class, "name"):
                                skill_name = getattr(skill_class, "name")
                            else:
                                skill_name = f"{skill_class.__name__} Skill"

                            self.log(f"Found Skill: {skill_name}")
                            if (
                                not hasattr(skill_class, "disabled")
                                or not getattr(skill_class, "disabled")
                                and hasattr(skill_class, "parse")
                            ):
                                settings.SKILLS_REGISTRY.append(
                                    skill_class(chat_session=self.chat_session, debug=self.debug)
                                )
                            else:
                                self.log(f"{skill_name} skill is disabled.")

    def respond(self, chat_query):
        message = None
        source = "ERROR"
        responded = False

        for skill in settings.SKILLS_REGISTRY:
            sc = skill
            try:
                responded = sc.parse(chat_query)
            except Exception as e:
                self.log(str(e))
            else:
                self.log(f"{skill}: {responded}")
                if not is_empty(responded):
                    return responded, str(skill)

        if is_empty(responded):
            completions = openai.Completion.create(
                engine="text-davinci-002",
                prompt=chat_query,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )

            message = completions.choices[0].text
            source = "ChatGPT"

            responded = True

            if responded:
                return message, source

        return responded
