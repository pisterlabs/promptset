import json
import logging

from .openai_client import openai, model

game_logger = logging.getLogger("game")
openai_logger = logging.getLogger("openai")

_messages = [
    {
        "role": "system",
        "content": """
You are a game story writer.
I will provide you with a JSON object describing a game and it's initial location.

The backstory must:
- Introduce the hero and the setting of the game
- Describe the hero, their personality, and their background
- Describe how the hero came to be in the starting location of the game.
- Provide any other relevant background information for the game
- Be written in an artistic and enigmatic way.
- Be 300-500 characters long."""
    },
    {
        "role": "user",
        "content": """{"title": {"THEME": "Medieval Quest", "THEME DESCRIPTION": "Embark on a journey through a world of knights, dragons and magic", "TITLE": "The Sword and the Serpent", "TITLE DESCRIPTION": "As the brave knight Percival, set out on a quest to retrieve a magical sword stolen by a fearsome dragon. Battle towering monsters, uncover dark secrets, and overcome perilous obstacles on your journey to restore peace to the realm and become a legendary hero."}, "location": "King's castle"}"""
    },
    {
        "role": "assistant",
        "content": """In a faraway land, where magic flowed like a river and dragons soared through the sky, there was a kingdom called Avalon. A land once teeming with life and prosperity, now shrouded in darkness and fear. The source of this darkness was a powerful dragon that had descended upon the kingdom, stealing the magical sword that had kept the realm at peace for generations.

The king and his council knew that they needed a hero, someone brave and valiant enough to face the dragon and retrieve the sword. They searched far and wide, but none were deemed worthy of the task until they came across a young knight named Percival. His courage and bravery were legendary, and his determination to protect the kingdom was unrivaled.

But Percival was not the king's first choice. The knight had come to the attention of the kingdom's council by chance. He had been on a mission, battling against a group of bandits who had been terrorizing a nearby village. During the mission, Percival's bravery and skill caught the attention of one of the kingdom's messengers who was there to deliver a message to the king.

Upon hearing of the young knight's exploits, the king summoned him to the castle, and the council saw in him the qualities of a true hero. They believed that only he could save the kingdom from the darkness that had befallen it. And so, the council chose Percival for the quest to retrieve the magical sword and restore peace to Avalon.

Percival knew that the task ahead would be difficult, but he was determined to succeed. He would need all his skills, wits, and determination to overcome the challenges that lay ahead. He knew that the fate of the kingdom rested on his shoulders, and he was ready to do whatever it took to retrieve the sword and save Avalon."""
    },
    {
        "role": "user",
        "content": """{'title': {'THEME': 'Lunar Landing', 'THEME DESCRIPTION': "Experience the dangers and thrills of exploring the Moon's surface, battling the unknown, and discovering the secrets of humanity's celestial neighbor", 'TITLE': 'Lunar Explorer: Lost in Space', 'TITLE DESCRIPTION': "As the intrepid astronaut Jane, land on the lunar surface in the midst of a dangerous mission to establish humanity's presence on the Moon. Face unexpected challenges, navigate the rough terrain, and uncover secrets that have lain hidden since the dawn of humanity's space age in your quest to survive and return home."}, 'location': 'Apollo launchpad'}"""
    },
    {
        "role": "assistant",
        "content": """As humanity sought to explore beyond the reaches of our planet, the Moon emerged as a symbol of possibility and adventure. The journey to establish humanity's presence on the lunar surface was a daunting task, requiring a team of intrepid astronauts with the skills and determination to undertake such an endeavor.

Among them was Jane, a seasoned astronaut with a distinguished record of space missions. She had been selected for this mission because of her bravery, expertise, and resourcefulness in the face of danger.

As the day of the launch approached, Jane felt a mix of excitement and trepidation. She knew that the mission was a challenging one, and that every detail would need to be carefully planned and executed to ensure success. She had trained for years for this moment, and was ready to face whatever challenges lay ahead.

As she walked towards the launchpad, her heart was pounding with anticipation. This was the moment she had been waiting for, the chance to leave her footprints on a new world and to help pave the way for future generations of explorers. She took a deep breath, and prepared herself for the journey ahead."""
    },
    {
        "role": "user",
        "content": """{'title': {'THEME': 'Haunted Mansion Mystery', 'THEME DESCRIPTION': 'Explore a creepy mansion, solve puzzles, and uncover dark secrets', 'TITLE': 'Echoes of the Past: The Ghostly Manor', 'TITLE DESCRIPTION': 'As paranormal investigator Sarah, explore a creepy mansion that has been abandoned for decades. Piece together clues left behind by the former residents, solve haunting puzzles, and use your wits to survive supernatural encounters as you uncover the dark secrets of the ghostly manor.'}, 'location': 'Gates of the mansion'}"""
    },
    {
        "role": "assistant",
        "content": """Amidst the dense forest, an abandoned mansion lies in ruin. It's a place where shadows dance and whispers float through the air, and where those who dare to enter are never quite the same when they leave.

Sarah, a seasoned paranormal investigator, has been summoned to uncover the truth behind the mansion's dark history. With her assistant and a groundskeeper in tow, she drives deep into the forest, the headlights of their vehicle the only illumination amidst the pitch-black night.

As they approach the mansion, the air grows thick with a palpable sense of foreboding. The rustling of leaves in the wind takes on an otherworldly quality, and the darkness seems to cling to them like a cloak.

Sarah leads her team towards the entrance, determined to unravel the secrets of this cursed mansion. But as they step out of their vehicle and the mansion's broken windows and crumbling walls loom in the distance, they can't shake the feeling that something is watching them from the shadows.

The whispers grow louder, the shadows more sinister. With a sense of trepidation and a hint of excitement, Sarah and her team prepare to face whatever lies within. But as they step towards the mansion, they can't help but wonder if they will ever truly leave."""
    },
    {
        "role": "user",
        "content": """{'title': {'THEME': '1940s Chicago Private Eye', 'THEME DESCRIPTION': 'Experience the gritty underworld of 1940s Chicago, solve crimes, and fight corruption', 'TITLE': 'No One to Trust', 'TITLE DESCRIPTION': 'As the hardboiled detective Jack Daniels, navigate the seedy underbelly of Chicago in the 1940s. Solve crimes, gather evidence, and outsmart corrupt officials and ruthless gangsters to uncover the truth behind a conspiracy that threatens the city itself. In a world where no one can be trusted, it’s up to you to find the justice that is buried deep in the shadows.'}, 'location': 'Jack's Office'}, {'id': 3, 'title': 'The Illegal Gambling Den', 'characters': ['Jack Daniels', 'Shady Gamblers', 'Thugs'], 'objective': 'Bust an illegal gambling den run by a corrupt official to gather intelligence on the murder case and expose corruption.', 'location': 'Secret Gambling Den'}, {'id': 4, 'title': 'The Mysterious Package', 'characters': ['Jack Daniels', 'Delivery Boy', 'The Law'], 'objective': 'Collect a mysterious package with evidence vital to the case, but evade corrupt officials and crooked cops.', 'location': 'Secret Drop-off Point'}, {'id': 5, 'title': 'The Final Confrontation', 'characters': ['Jack Daniels', 'The Killer', 'Corrupt Official'], 'objective': 'Uncover the truth behind the conspiracy linking corrupt officials and ruthless gangsters, and bring the killer to justice.', 'location': "The Killer's Hideout"}]}"""
    },
    {
        "role": "user",
        "content": """The city is dark and gritty, like a well-worn fedora on a private eye. Jack Daniels is one such gumshoe, wandering the underbelly of 1940s Chicago. He knows the city's secrets and it knows his. A weary cynic, Jack is haunted by his past, but driven to right the wrongs of the present.

How did Jack end up here, a lone figure in a dingy office, surrounded by empty bottles and old case files? Perhaps it was fate, or perhaps it was something more sinister. The truth is shrouded in mystery, like the smoky haze of a jazz club.

The city hums with a palpable tension, like the warning growl of a tommy gun. Jack knows that something is brewing in the shadows, something big. He can feel it in his bones, the way the air shifts when a storm is coming.

As Jack stares out the window, his thoughts drift to the past. A case gone wrong, a betrayal, a loss. But he can't dwell on it now. There's work to be done. And Jack Daniels is a man who always gets the job done."""
    }
]


def generate_backstory(game: dict):
    game_logger.info(f"Generating backstory for game: {game}")
    response = openai.ChatCompletion.create(
        model=model,
        messages=_messages + [{"role": "user", "content": f"{game}"}],
    )
    openai_logger.info(f"OpenAI response: {json.dumps(response, indent=2)}")
    backstory = response['choices'][0]['message']['content']
    return backstory
