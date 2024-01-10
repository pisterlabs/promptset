import json
from langchain.llms import HuggingFacePipeline
from marshmallow import ValidationError

from environment.types import CrimeSceneMap
from llm.chains.killer_chain import KillerChain
from llm.chains.rooms_chain import RoomsChain
from llm.chains.suspect_chain import SuspectChain
from llm.chains.victim_chain import VictimChain
from llm.marshmallow.schemas.story import StorySchema


class StoryGenerator:
    def __init__(self, crime_scene_map: CrimeSceneMap, llm_pipeline) -> None:
        self._crime_scene_map = crime_scene_map
        self._llm_pipeline = llm_pipeline

    def create_new_story(self, theme: str, dummy: bool = False) -> StorySchema:
        if dummy:
            with open("./llm-dungeon-adventures/data/dummy.json", "r") as f:
                return json.load(f)

        victim = VictimChain(self._llm_pipeline).create(theme)
        killer = KillerChain(self._llm_pipeline).create(theme, victim)
        suspects = SuspectChain(self._llm_pipeline).create(theme, victim)
        rooms, suspects_positions = RoomsChain(self._llm_pipeline, self._crime_scene_map).create(
            theme, victim, suspects + [killer]
        )

        try:
            return StorySchema().load(
                {
                    "theme": theme,
                    "victim": victim,
                    "killer": killer,
                    "suspects": suspects,
                    "rooms": rooms,
                    "suspects_positions": suspects_positions,
                }
            )
        except ValidationError as err:
            print(err.messages)
            return None
