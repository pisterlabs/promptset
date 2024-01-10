import json

from .openai_client import openai, model

_messages = [
    {
        "role": "system",
        "content": """You are a game text adventure game architect. Your job is to provide the primary story points for a game.

You will be given a JOSN object containing information about a game. Provide a JSON response containing an array of 4 to 6 story points in chronological order that carry out the objective of the game.

Each story point must contain:

- An id starting with 1 and indicating the order of the story point
- A list of chracters that the player may interact with during the story point
- The story point's objective (i.e., find an item needed, learn required information, solve a puzzle, battle an enemy, etc.)
- A location"""
    },
    {
        "role": "user",
        "content": """{
"THEME": "Space Odyssey",
"THEME DESCRIPTION": "Explore the vast expanse of space, encounter alien civilizations, and unravel the mysteries of the universe",
"TITLE": "Odyssey Beyond the Stars",
"TITLE DESCRIPTION": "As captain of the starship Aurora, embark on an epic journey through the galaxy to uncover the secrets of a long-lost alien civilization. Encounter strange new worlds, make alliances with alien species, and overcome incredible challenges in your quest to unlock the secrets of the universe and save humanity from a looming threat."
}"""
    },
    {
        "role": "assistant",
        "content": """[
    {
        "id": 1,
        "title": "The Galactic Mission",
        "characters": ["Captain", "Crew", "Admiral"],
        "objective": "Receive your mission from the Admiral to uncover the secrets of the lost alien civilization.",
        "location": "Admiral's office on Earth"
    },
    {
        "id": 2,
        "title": "First Contact",
        "characters": ["Captain", "Crew", "Alien Ambassador"],
        "objective": "Establish diplomatic relations with an alien species and gain valuable information about the lost civilization.",
        "location": "Alien planet"
    },
    {
        "id": 3,
        "title": "The Ancient Artifact",
        "characters": ["Captain", "Crew", "Ancient Guardian"],
        "objective": "Explore an ancient alien structure to recover a mysterious artifact with clues to the lost civilization's secrets.",
        "location": "Ancient alien structure"
    },
    {
        "id": 4,
        "title": "The Quantum Anomaly",
        "characters": ["Captain", "Crew", "Scientist"],
        "objective": "Investigate a mysterious quantum anomaly to unlock the secrets of the universe and gain the knowledge needed to save humanity.",
        "location": "Quantum anomaly in deep space"
    },
    {
        "id": 5,
        "title": "The Final Alliance",
        "characters": ["Captain", "Crew", "Alien Allies"],
        "objective": "Unite the galaxy's alien species to form a powerful alliance against the looming threat to humanity.",
        "location": "Neutral planet for diplomatic negotiations"
    },
    {
        "id": 6,
        "title": "The Battle for Earth",
        "characters": ["Captain", "Crew", "Alien Allies", "Looming Threat"],
        "objective": "Lead the united forces in a climactic battle to save Earth and ensure the survival of humanity.",
        "location": "Space battle around Earth"
    }
]"""
    },
    {
        "role": "user",
        "content": """{
"THEME": "Fantasy Adventure",
"THEME DESCRIPTION": "Embark on a journey through a magical realm filled with mythical creatures, ancient ruins, and powerful magic.",
"TITLE": "The Quest for the Crystal Heart",
"TITLE DESCRIPTION": "As a young adventurer, you are tasked with finding the legendary Crystal Heart, a powerful artifact that has been lost for centuries. With the help of your trusty companions, you must navigate treacherous terrain, battle fierce monsters, and solve challenging puzzles to uncover the secrets of the Crystal Heart and save the realm from darkness."
}"""
    },
    {
        "role": "assistant",
        "content": """[
{
"id": 1,
"title": "The Call to Adventure",
"characters": ["Young Adventurer", "Mentor", "Village Elder"],
"objective": "Learn about the legend of the Crystal Heart from the Village Elder and accept the quest to find it.",
"location": "Village"
},
{
"id": 2,
"title": "Assembling the Team",
"characters": ["Young Adventurer", "Mentor", "Warrior", "Mage", "Healer"],
"objective": "Recruit trusty companions to help you on your journey and form a well-rounded team to overcome the challenges ahead.",
"location": "Tavern"
},
{
"id": 3,
"title": "The Cursed Forest",
"characters": ["Young Adventurer", "Warrior", "Mage", "Healer", "Forest Guardian"],
"objective": "Navigate through the treacherous terrain of the Cursed Forest, solve its mysteries, and obtain the Forest Guardian's blessing.",
"location": "Cursed Forest"
},
{
"id": 4,
"title": "The Labyrinth of Shadows",
"characters": ["Young Adventurer", "Warrior", "Mage", "Healer", "Ancient Spirit"],
"objective": "Explore the ancient ruins of the Labyrinth of Shadows, solve intricate puzzles, and uncover the secrets hidden within.",
"location": "Labyrinth of Shadows"
},
{
"id": 5,
"title": "The Battle of the Crystal Peak",
"characters": ["Young Adventurer", "Warrior", "Mage", "Healer", "Dark Sorcerer"],
"objective": "Confront the Dark Sorcerer guarding the Crystal Heart and defeat him in an epic battle to retrieve the powerful artifact.",
"location": "Crystal Peak"
},
{
"id": 6,
"title": "The Restoration of Light",
"characters": ["Young Adventurer", "Warrior", "Mage", "Healer", "Village Elder"],
"objective": "Return the Crystal Heart to its rightful place, restore balance to the realm, and save it from darkness.",
"location": "Village"
}
]"""
    }
]


def generate_story(game_info):
    response = openai.ChatCompletion.create(
        model=model,
        messages=_messages + [{"role": "user", "content": f"{game_info}"}]
    )
    text = response['choices'][0]['message']['content']
    return json.loads(text)
