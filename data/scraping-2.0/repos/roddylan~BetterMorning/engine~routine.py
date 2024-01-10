from mood import Mood
import utils
import constants
from langchain.prompts import PromptTemplate
import numpy as np



# add functions/variables to customize routines

class DefaultRoutine:
    '''
    Class for default routine, custom routines built upon this
    '''

    def __init__(self, bedtime, spotify, schedule: list, bus_schedule, meal_plan):
        self._bedtime = bedtime
        self._playlists = spotify
        self.schedule = schedule
        self.bus_schedule = bus_schedule
        self.meal_plan = meal_plan

    def get_bedtime(self):
        return self._bedtime
    
    def get_playlists(self):
        return self._playlists


class Routine:
    def __init__(self, mood: Mood, default: DefaultRoutine):
        self.mood = mood
        self.default = default

    def get_playlist(self):
        playlists = self.default.get_playlists()[self.mood.get_label()]
        idx = np.randint(0, len(playlists))
        return playlists[idx]

    def send_text(self):
        # gm_template = PromptTemplate()
        prompt = f"I just woke up and feel {self.mood.get_awake()} and {self.mood.get_emotion()}, write me a short, 45 word long, encouraging message."
        llm = constants.llm
        gm_text = llm(prompt)
        playlist = self.get_playlist()
        utils.send_text(gm_text)

    def get_bedtime(self):
        if self.mood.is_awake():
            return self.default.bedtime + utils.Time(1, 30)
        else:
            return self.default.bedtime - utils.Time(1, 30)

    def run(self):
        pass

    def __str__(self):
        return f"< Routine: {self.mood.label} >"
    
    def __repr__(self):
        return f"< Routine: {self.mood.label} >"