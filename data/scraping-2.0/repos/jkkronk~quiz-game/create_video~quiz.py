from pydantic import BaseModel, Field
from openai import OpenAI
import instructor
import json
import random

class QuizHost():
    intro: str = Field(..., description="The introduction of the quiz.")
    outro: str = Field(..., description="The outro of the quiz.")

    #init
    def __init__(self, intro, outro):
        self.intro = intro
        self.outro = outro

class QuizClues(BaseModel):
    clues: list[str] = Field(..., description="A list of size 5 with clues for the quiz.")
    explanations: list[str] = Field(..., description="A list of size 5 with explanations for the clues.")

    def clear_city(self):
        self.clues = [clue.replace("Zurich", "the city") for clue in self.clues]

    def get_clue(self, round: int) -> str:
        return self.clues[round]

    def get_explanation(self, round: int) -> str:
        return self.explanations[round]

    def get_all_clues(self):
        return "\n".join(self.clues)

    def get_all_explanation(self) -> str:
        return "\n".join(self.explanations)

    def save(self, city, file_path: str):
        data = {
            "city": city,
            "clues": self.clues,
            "explanations": self.explanations
        }
        with open(file_path, 'w') as file:
            json.dump(data, file)

    @classmethod
    def open(cls, file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return cls(clues=data['clues'], explanations=data['explanations'])

def random_destination() -> str:
    # open the cities text file and pick a random city
    # return the city
    path_to_cities = "static/cities.txt"

    # Opening the file
    with open(path_to_cities, 'r') as file:
        cities_text = file.read()

    # Splitting the text into a list of cities
    cities_list = cities_text.split(',')

    # Selecting a random city from the list
    random_city = random.choice(cities_list)

    return random_city.replace("\n", "")

def create_quiz(city:str, openai_api_key="") -> QuizClues:
    if openai_api_key == "":
        client = instructor.patch(OpenAI())
    else:
        client = instructor.patch(OpenAI(api_key=openai_api_key))

    prompt = f"""
                You are a quiz host and you are hosting a quiz where the answer is {city}. You are suppose to come up
                with 5 clues for the city. Each clue should be easier and easier. In the beginning it 
                shall be very hard. But in the end it shall be very easy. 
                
                Each clue should be a couple of sentences long. 
                The clues should be written in English. 
                The clues could be on historic facts, famous persons, famous buildings, famous events, famous food, 
                famous drinks, famous music, famous art, famous sports, famous games, famous movies, famous books, from 
                {city}. 
                Each clue should end with "..."
                The clues should be humorous and engaging. It can be a bit hard to guess the answer and there shall be
                word plays, rimes and puns.
                
                Additionally, add a short explanation for each clue.
                
                An example for New York could be:
                >>First Clue: We leave religious cheese and head north towards great fruit. Liberal Dutch have a historical stamp in our five-star Jorvik...
                >>Second Clue: The theatrical stage of the masculine headdress lies on the broad road. Reuterswärd's revolver pacifies Moon's headquarters...
                >>Third Clue: In the center of our new world city is a square garden. Temperate climate notwithstanding, you can skate at Rockefeller but scratch the squares first...
                >>Fourth Clue: At our entrance we are greeted by freedom unless you land at JFK. King Kong climbed here on an empirical building, but the economy was managed from Vallgatan...
                >>Last Clue: September 11 will always be remembered in the United States' largest city. Well, you know the answer...
                
                Another example for Bejing/Peking could be:
                >>First Clue: We travel from one "heavenly" city to another in a country where, in other words, you can give clues. Old summer palace was destroyed at our destination by English troops during war in 1860. Other summer palace is also great attraction here in the city...
                >>Second Clue: Our many-thousand-year-old city has historically been replaced with its southern part. The short-lived lightning record was broken here and happened in just 9.69 seconds. Birdsnest is another clue...
                >>Third Clue: The Swedish evergreen tree is also said to be extremely popular here. 08 and 22, well then, world olympians competed in our destination...
                >>Fourth Clue: Our semi-forbidden city is the country's second after shark city and is a possible final destination for the Trans-Siberian journey. When you arrive, you will probably be greeted with "ni hao". Maybe you will also be treated to duck...
                >>Last Clue:  Now we have arrived in China's capital, where the English king is pointed...
                """


    clues: QuizClues = client.chat.completions.create(
        model="gpt-4",
        response_model=QuizClues,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_retries=2,
    )

    return clues
