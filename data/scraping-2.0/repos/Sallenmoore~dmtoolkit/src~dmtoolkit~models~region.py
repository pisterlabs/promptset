import json
import random

from autonomous import log
from autonomous.ai import OpenAI

from dmtoolkit.models import City, Encounter, Faction, Location
from dmtoolkit.models.ttrpgobject import TTRPGObject


class Region(TTRPGObject):
    
    cities:list = []
    locations:list = []
    factions:list = []
    encounters:list = []

    _environments = [
        "coastal",
        "mountainous",
        "desert",
        "forest",
        "plains",
        "swamp",
        "frozen",
        "underground",
    ]

    @property
    def funcobj(self):
        return {
            "name": "generate_region",
            "description": "creates Region data object",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The region's name",
                    },
                    "desc": {
                        "type": "string",
                        "description": "A physical description of the region",
                    },
                    "backstory": {
                        "type": "string",
                        "description": "A brief history of the region",
                    },
                },
            },
        }

    @classmethod
    def generate(cls, world, description="magic exists"):
        trait = [random.choice(cls._environments)]
        primer = """
        As an expert AI in fictional Worldbuilding, generate a fictional region complete with a name and description.
        """
        prompt = f"Generate a fictional {world.genre} region with a {trait} setting within a world with the following description: {description or world.desc}. Write a detailed description containing an unusual, wonderful, OR sinister secret from the region's history."
        obj_data = super().generate(primer, prompt)
        obj_data |= {"world": world, "traits": trait}
        obj = cls(**obj_data)
        obj.save()
        return obj

    def get_image_prompt(self):
        prompt = f"An aerial pictoral map illustration of the fictional region {self.name} with the following description: {self.desc}."
        if self.cities:
            cities = ",".join([c.name for c in self.cities])
            prompt += f"The region contains the following cities: {cities}."
        return prompt

    def create_cities(self, n=1):
        for _ in range(n):
            self.cities.append(City.generate(world=self.world, description=self.desc))
        if self.factions:
            for city in self.cities:
                city.factions = self.factions
        self.save()
        return self.cities

    def create_locations(self, n=1):
        ltypes = [
            "cave",
            "ruin",
            "temple",
            "fortress",
            "tower",
            "swamp",
            "forest",
            "mountain",
        ]
        description = f"An explorable {random.choice(ltypes)} with secrets and story threads in a region described as follows: {self.desc}."
        for _ in range(n):
            l = Location.generate(world=self.world, description=description)
            self.locations.append(l)
        self.save()
        return self.locations

    def create_factions(self, n=1):
        for _ in range(n):
            self.factions.append(
                Faction.generate(world=self.world, description=self.desc)
            )
        if self.cities:
            for city in self.cities:
                city.factions = self.factions
        self.save()
        return self.factions

    def create_encounter(self, num_players=5, level=1):
        for _ in range(n):
            self.encounters.append(
                Encounter.generate(world=self.world, num_players=5, level=1)
            )
        self.save()
        return self.encounters

    def page_data(self, root_path="ttrpg"):
        cities = [f"[{r.name}]({r.wiki_path})" for r in self.cities]
        locations = [f"[{r.name}]({r.wiki_path})" for r in self.locations]
        factions = [f"[{r.name}]({r.wiki_path})" for r in self.factions]
        encounters = {r.name: r.page_data() for r in self.encounters}
        return {
            "Cities": cities,
            "Locations": locations,
            "Factions": factions,
            "Encounters": encounters,
        }

    def canonize(self, api=None, root_path="ttrpg"):
        super().canonize(api, root_path)
        for f in [*self.factions, *self.locations, *self.cities]:
            f.canonize(api=api, root_path=root_path)
        self.save()
