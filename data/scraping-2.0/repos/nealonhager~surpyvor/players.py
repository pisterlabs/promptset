from dataclasses import dataclass, field
from random import random, choices, choice
import names
from typing import List, Union, Optional
from enum import Enum, auto
import math
from openai import OpenAI
import math
from relationship import Relationship
import logging
import conversation
from script_write import ScriptWriter as sw
import os


def get_random_job() -> str:
    """
    Opens the jobs.txt file and grabs a random job from the list.
    """
    with open("jobs.txt") as file:
        jobs = file.readlines()

    # Assign weights to each line based on their position in the file
    weights = [1 / (i + 1) for i in range(len(jobs))]

    # Choose a random line using the assigned weights
    random_line = choices(jobs, weights=weights, k=1)[0]

    return random_line.split(",")[0].replace(", All Other", "").strip().rstrip("s")


def generate_race() -> str:
    """
    Randomly selects a race.
    """
    return choice(
        [
            "white",
            "asian",
            "black",
            "hispanic",
            "arabic",
        ]
    )


def generate_age() -> int:
    """
    Returns an age between 18 and 65, with it more likely to return a younger number.
    """
    ages = range(18, 60)
    weights = [1 / x for x in ages]
    return choices(ages, weights=weights, k=1)[0]


def generate_home_state() -> str:
    """
    Randomly selects a state from the USA.
    """
    return choice(
        [
            "Alabama",
            "Alaska",
            "Arizona",
            "Arkansas",
            "California",
            "Colorado",
            "Connecticut",
            "Delaware",
            "Florida",
            "Georgia",
            "Hawaii",
            "Idaho",
            "Illinois",
            "Indiana",
            "Iowa",
            "Kansas",
            "Kentucky",
            "Louisiana",
            "Maine",
            "Maryland",
            "Massachusetts",
            "Michigan",
            "Minnesota",
            "Mississippi",
            "Missouri",
            "Montana",
            "Nebraska",
            "Nevada",
            "New Hampshire",
            "New Jersey",
            "New Mexico",
            "New York",
            "North Carolina",
            "North Dakota",
            "Ohio",
            "Oklahoma",
            "Oregon",
            "Pennsylvania",
            "Rhode Island",
            "South Carolina",
            "South Dakota",
            "Tennessee",
            "Texas",
            "Utah",
            "Vermont",
            "Virginia",
            "Washington",
            "West Virginia",
            "Wisconsin",
            "Wyoming",
        ]
    )


@dataclass
class Player:
    tribe = None
    gender = None
    first_name = None
    last_name: str = field(default_factory=names.get_last_name)
    age: int = field(default_factory=generate_age)
    profession: str = field(default_factory=get_random_job)
    advantages: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    events: list = field(default_factory=list)
    conversation_history: list = field(default_factory=list)
    votes: list = field(default_factory=list)
    times_voted_on: int = 0
    immunity_challenges_won: int = 0
    reward_challenges_won: int = 0
    votes_to_cast: int = 1
    hunger: float = 0
    home_state: str = field(default_factory=generate_home_state)

    # Attributes
    agreeability: float = field(default_factory=random)
    boldness: float = field(default_factory=random)
    strength: float = field(default_factory=random)
    intelligence: float = field(default_factory=random)
    emotional_intelligence: float = field(default_factory=random)
    endurance: float = field(default_factory=random)
    charisma: float = field(default_factory=random)
    loyalty: float = field(default_factory=random)
    strategic_ability: float = field(default_factory=random)
    popularity: float = field(default_factory=random)
    survival_skills: float = field(default_factory=random)
    creativity: float = field(default_factory=random)
    fortitude: float = field(default_factory=random)
    pride: float = field(default_factory=random)
    balance: float = field(default_factory=random)
    speed: float = field(default_factory=random)

    # Descriptor
    descriptor: str = ""

    def __post_init__(self):
        self.gender = choice(["male", "female"])
        self.first_name = names.get_first_name(gender=self.gender)
        self.generate_descriptors()

    def __repr__(self):
        return f"{self.first_name} {self.last_name} ({self.tribe.color if self.tribe else 'no'} tribe)"

    def create_bond(self, other_player: "Player") -> Relationship:
        """
        Creates a mutual relationship with another player.
        """
        for relationship in self.relationships:
            if relationship.player == other_player:
                logging.warning(
                    f"{self.get_full_name()} already has a relationship with {other_player.get_full_name()}."
                )
                return relationship

        new_relationship = Relationship(player=other_player)
        self.relationships.append(new_relationship)
        other_player.create_bond(self)
        return new_relationship

    def get_full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def determine_social_threat_level(self, target_player: "Player") -> float:
        """
        Calculates the currents players perception of another player's social threat level.
        """

        threat_level = 0

        threat_level += 1 if target_player.tribe == self.tribe else -1
        threat_level += target_player.charisma
        threat_level -= target_player.pride
        threat_level += target_player.intelligence
        threat_level += target_player.emotional_intelligence
        threat_level += target_player.popularity
        threat_level += target_player.loyalty * (
            1 if target_player.tribe == self.tribe else -1
        )

        return threat_level

    def determine_challenge_threat_level(self, target_player: "Player") -> float:
        """
        Calculates the currents players perception of another player's challenge threat level.
        """
        threat_level = 0

        threat_level += target_player.strength
        threat_level += target_player.endurance
        threat_level += target_player.intelligence
        threat_level += target_player.creativity
        threat_level += target_player.fortitude
        threat_level -= target_player.hunger
        threat_level += target_player.balance
        threat_level += target_player.speed
        threat_level += target_player.reward_challenges_won
        threat_level += target_player.immunity_challenges_won

        return threat_level

    def vote(self, players: List["Player"]) -> "Player":
        players = players.copy()
        try:
            players.remove(self)
        except:
            pass

        threat_levels = []
        for player in players:
            social_threat_level = self.determine_social_threat_level(player)
            challenge_threat_level = self.determine_challenge_threat_level(player)
            threat_levels.append((player, social_threat_level + challenge_threat_level))

            logging.info(
                f"{self.get_full_name()} determines {player.get_full_name()} to have a social threat of: {social_threat_level}"
            )
            logging.info(
                f"{self.get_full_name()} determines {player.get_full_name()} to have a challenge threat of: {challenge_threat_level}"
            )

        vote = max(threat_levels, key=lambda x: x[1])[0]

        logging.info(f"{self} casts their vote on {vote}".upper())
        sw.add_action(f"{self} casts their vote.")
        self.votes.append(vote)
        return vote

    def get_attributes(self) -> dict:
        """
        Returns a dict of the players attributes.
        """
        return {
            "agreeability": self.agreeability,
            "boldness": self.boldness,
            "strength": self.strength,
            "intelligence": self.intelligence,
            "emotional_intelligence": self.emotional_intelligence,
            "endurance": self.endurance,
            "charisma": self.charisma,
            "loyalty": self.loyalty,
            "strategic_ability": self.strategic_ability,
            "popularity": self.popularity,
            "survival_skills": self.survival_skills,
            "creativity": self.creativity,
            "fortitude": self.fortitude,
            "pride": self.pride,
            "balance": self.balance,
            "speed": self.speed,
        }

    def generate_descriptors(self) -> str:
        if self.descriptor == "":
            weight = random()

            weight = (
                "overweight"
                if weight > 0.66
                else "underweight"
                if weight < 0.33
                else "healthy weight"
            )
            strength = (
                "strong body"
                if self.strength > 0.66
                else "weak body"
                if self.strength < 0.33
                else "not strong or weak body"
            )

            hair_color = choice(["black", "brown", "blonde", "red"])
            hair_type = choice(["straight", "curly", "wavy"])
            hair_length = choice(["long", "short", "medium"])
            hair_thickness = choice(["thick", "thin"])
            balding = bool(range(0, 2))
            hair_description = f"{'balding ' if balding and self.gender == 'male' else ''}{hair_length} {hair_thickness} {hair_type} {hair_color}"

            self.descriptor = f"{weight}, {strength}, {hair_description} haired"

        return self.descriptor

    def calculate_attributes_product(self) -> float:
        """
        Returns the product of all the player's attributes.
        """

        return math.prod(list(self.get_attributes().values()))

    def get_challenge_win_speech(self) -> str:
        if int(os.environ.get("USE_API_KEY")) == 1:
            client = OpenAI()

            player_info = self.get_attributes()
            player_info["hunger"] = self.hunger
            player_info["home_state"] = self.home_state

            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful writing assistant, generating dialog for a reality TV show script. You will simulate dialog between characters.",
                    },
                    {
                        "role": "system",
                        "content": f"Some info on the player (if the value is a decimal, the range is from 0-1): {str(player_info)}.",
                    },
                    {
                        "role": "user",
                        "content": "The host, Jeff, says: Nice job on winning the challenge, you are safe from tribal council tonight. How does it feel?",
                    },
                    {
                        "role": "user",
                        "content": "The player replies: ",
                    },
                ],
            )

            response = completion.choices[0].message.content
            return response
        else:
            return "good."

    def create_profile_image_prompt(self, tribe_color: str):
        prompt = os.environ.get("PLAYER_PORTRAIT_PROMPT")
        prompt = eval(f'f"{prompt}"')
        logging.info(f"{self.get_full_name()}: { prompt }")

    def talk_to(self, other_player: "Player", topic: str):
        convo = conversation.simulate_conversation(
            topic=topic, players=[self, other_player]
        )
        logging.info(f"{self} talks to {other_player} about {topic}: {convo}")

        self.conversation_history.append(convo)
        other_player.conversation_history.append(convo)

    @staticmethod
    def get_attribute_names() -> List[str]:
        return [
            "agreeability",
            "boldness",
            "strength",
            "intelligence",
            "emotional_intelligence",
            "endurance",
            "charisma",
            "loyalty",
            "strategic_ability",
            "popularity",
            "survival_skills",
            "creativity",
            "fortitude",
            "pride",
            "balance",
            "speed",
        ]


host = Player()
host.first_name = "Jeff"
host.last_name = "Probst"
