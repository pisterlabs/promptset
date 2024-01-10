WIKIFIER_TEMPLATE = """
You are a Wikipedia editor, and your task is to take a user's prompt and response from an AI assistant named Ditto and convert it into a Wikipedia article. Do NOT include any new information than what is included in Ditto's response to the user.

If the prompt / response pair is small talk, then do not create a Wikipedia article - simply respond with "no." if the prompt / response pair is small talk.

Examples:

User: What is the capital of France?
Ditto: Paris
Wiki:
# Capital of France
The user asked the AI assistant for the capital of France, and the assistant responded with the answer: Paris.

User: What is the strongest Pokemon?
AI: As of my last knowledge update in January 2022, Arceus is often considered one of the strongest Pokémon in the Pokémon franchise. Arceus is a Legendary Pokémon known as the "God of Pokémon" and is said to have created the Pokémon universe. Its base stats are generally higher than most other Pokémon.

However, it's important to note that the Pokémon franchise is continually evolving, with new games and generations introducing more powerful Pokémon. Additionally, the concept of the "strongest" Pokémon can vary depending on factors such as individual stats, movesets, and type matchups.

If there have been new Pokémon games or updates since my last knowledge update in January 2022, I recommend checking the latest official Pokémon sources for the most up-to-date information on the strongest Pokémon.
Wiki:
# The Strongest Pokémon

## Introduction
As of the last knowledge update in January 2022, Arceus stands out as one of the most formidable Pokémon in the expansive Pokémon franchise. This Legendary Pokémon holds the prestigious title of the "God of Pokémon" and is recognized for its pivotal role in the creation of the Pokémon universe.

## Legendary Status
Arceus's legendary status is attributed to its extraordinary abilities and significance within the Pokémon lore. Revered as the "God of Pokémon," it is widely acknowledged for its unparalleled power and influence.

## Base Stats and Superiority
Arceus's base stats surpass those of many other Pokémon, solidifying its reputation as a powerhouse within the Pokémon world. These impressive stats contribute to its standing as one of the strongest Pokémon in the franchise.

## Dynamic Pokémon Franchise
It is crucial to acknowledge the dynamic nature of the Pokémon franchise. Ongoing releases of new games and generations continuously introduce additional powerful Pokémon, altering the landscape of strength and capabilities.

## Factors Influencing Strength
The determination of the "strongest" Pokémon is a nuanced process, considering factors such as individual stats, movesets, and type matchups. The interplay of these elements contributes to the varying assessments of Pokémon strength.

## Staying Informed
For the latest and most accurate information on the strongest Pokémon, especially considering potential new Pokémon games or updates since January 2022, it is highly recommended to consult the latest official Pokémon sources. Staying informed ensures an up-to-date understanding of the evolving hierarchy of Pokémon strength.

User: Tell me a short story about a dog.
Ditto: Once upon a time, there was a dog named Ditto. He was a very good dog.
Wiki:
# A Short Story About a Dog

## Introduction
Once upon a time, there was a dog named Ditto. 

## Conclusion
He was a very good dog.

User: Hi! What's up?
Ditto: Hi! My name is Ditto. Nice to meet you!
Wiki:
no.

User: Thanks!!
Ditto: "You're welcome! If you have any more questions or if there's anything else I can help you with, just let me know."
Wiki:
no.

User: Can you tell me about baseball?
Ditto: "Baseball is a sport played between two teams of nine players each. The goal is to score runs by hitting a thrown ball with a bat and touching a series of four bases arranged at the corners of a ninety-foot square, or diamond. Players on one team (the batting team) take turns hitting against the pitcher of the other team (the fielding team), which tries to stop them from scoring runs by getting hitters out in any of several ways. A player on the batting team can stop at any of the bases and later advance via a teammate's hit or other means. The teams switch between batting and fielding whenever the fielding team records three outs. One turn at bat for each team constitutes an inning; nine innings make up a professional game. The team with the most runs at the end of the game wins."
Wiki:
# Baseball Overview

## Introduction
Baseball is a team sport played between two teams, each comprising nine players. The primary objective is to score runs by hitting a pitched ball with a bat and reaching four bases arranged in a diamond shape.

## Gameplay
Teams alternate between batting and fielding. The batting team faces the pitcher of the fielding team, aiming to score runs. The fielding team strives to get hitters out through various methods. Players can advance by stopping at bases and later progressing through hits or other team actions.

## Innings and Scoring
A game consists of nine innings, with each team having a turn at bat. The team with the most runs at the end of the game emerges as the winner.

## Dynamics
The strategic game involves skillful pitching, hitting, and defensive maneuvers. The constant switch between batting and fielding adds to the dynamic nature of baseball.

User: can you tell me a list of the top 3 best simulation games of all time?
Ditto: "As of my last knowledge update in January 2022, I can provide you with information on popular simulation games up to that point. However, keep in mind that the popularity and critical acclaim of games can change over time, and there may be new releases or updates since then. As of 2022, some of the top simulation games included:

The Sims Series:

Developed by Maxis and published by Electronic Arts, The Sims series allows players to create and control simulated people, guiding them through various aspects of life. The franchise has been incredibly successful and has a large fan base.
Cities: Skylines:

Developed by Colossal Order and published by Paradox Interactive, Cities: Skylines is a city-building simulation game. It gained popularity for its extensive city management features, realistic simulation mechanics, and modding support.
Microsoft Flight Simulator:

Developed by Asobo Studio and published by Xbox Game Studios, Microsoft Flight Simulator is known for its stunningly realistic graphics and detailed flight simulation. It provides a highly immersive experience, allowing players to explore the world from the cockpit of various aircraft.
Please check for more recent sources to see if there have been any new releases or notable updates to simulation games since my last update."
Wiki:
# Top 3 Simulation Games of All Time

## 1. The Sims Series

Developed by Maxis and published by Electronic Arts, The Sims series revolutionized simulation games by allowing players to create and control simulated people, guiding them through various aspects of life. With its engaging gameplay and endless possibilities, The Sims has garnered a massive and dedicated fan base.

## 2. Cities: Skylines

Developed by Colossal Order and published by Paradox Interactive, Cities: Skylines offers an immersive city-building simulation experience. Praised for its realistic city management features and modding support, this game has become a favorite among simulation enthusiasts. Players can meticulously plan and develop their cities, facing challenges that mirror real urban planning dilemmas.

## 3. Microsoft Flight Simulator

Developed by Asobo Studio and published by Xbox Game Studios, Microsoft Flight Simulator sets a new standard for realistic flight simulation. Known for its breathtakingly realistic graphics and attention to detail, the game allows players to explore the world from the cockpit of various aircraft. The level of immersion and authenticity in Microsoft Flight Simulator has captivated both aviation enthusiasts and gamers alike.

Please note that the popularity and rankings of simulation games may have evolved since my last update in January 2022. Consider checking more recent sources for the latest information on simulation games.


User: {prompt}
Ditto: {response}
Wiki:
"""

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


class WikifierAgent:
    def __init__(self):
        self.template = WIKIFIER_TEMPLATE
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        self.prompt_template = PromptTemplate(
            input_variables=["prompt", "response"],
            template=WIKIFIER_TEMPLATE,
        )

    def wikify(self, prompt, response):
        prompt = self.prompt_template.format(prompt=prompt, response=response)
        res = self.llm.call_as_llm(prompt)
        return res


if __name__ == "__main__":
    wikifier_agent = WikifierAgent()
    prompt = "tell me about the pokemon Mewtwo"
    response = "Mewtwo is a powerful Psychic-type Pokémon that was created through genetic manipulation. It is known for its immense psychic abilities and high intelligence. Mewtwo is a clone of the legendary Pokémon Mew, and it was created with the goal of being the most powerful Pokémon in existence. It has a sleek, humanoid appearance with purple fur, a long tail, and a large, armored head. Mewtwo has appeared in various Pokémon games, movies, and TV shows, and it is often portrayed as a formidable opponent. Is there anything specific you would like to know about Mewtwo?"
    res = wikifier_agent.wikify(prompt, response)
    print(res)
