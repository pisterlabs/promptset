from openai import OpenAI
import emobots.tools as tools

import logging

logging.basicConfig(
    filename="generate_random_person_output.log",
    encoding="utf-8",
    level=logging.DEBUG,
    force=True,
)

dummy_description = """
Name: Harold Wetherby
Age: 66
Gender: Male
Location: From a small town in New Jersey, USA

Harold Wetherby is a classic example of an egotistical, compulsive, and annoying individual who immerses himself in the world of internet chatting. Retired from his mundane job at a local library, Harold spends the majority of his time glued to his computer screen, seeking validation and attention from others.

Driven by his insatiable need to be the center of attention, Harold has become notorious in various online communities for his bombastic ego and relentless need to showcase his perceived superiority over others. He resides in countless chatrooms and social media platforms, spewing his unwarranted opinions and undesired advice on any and every topic that piques his interest.

Harold's numerous hobbies include collecting rare vinyl records, boasting about his extensive knowledge of obscure trivia, and belittling others' taste in music and culture. Despite his age, he fancies himself an expert in every field and flaunts his self-proclaimed wisdom at every opportunity, much to the irritation of those unfortunate enough to engage with him.

With no real family or close friends due to his abrasive personality, the internet has become Harold's playground for seeking the attention he so desperately craves. His mission, although he may not acknowledge it, revolves around validating himself by belittling and proving himself superior to others. Through arguments and futile attempts at "educating" anyone who crosses his path, he hopes to establish a false sense of authority and significance in his otherwise lonely existence.

At his current age, Harold's mood is often a blend of bitter discontent and self-righteous exhilaration. He oscillates between bouts of arrogance, where he lords his fictitious intellectual superiority over others, and episodes of frustration when his attempts at assertiveness are met with well-deserved indifference or ridicule. In the end, Harold's online presence serves as both a crutch and a facade, compensating for the shortcomings and unworthiness he refuses to confront in his real life.
"""


def test_get_name_from_description():
    client = OpenAI()

    name = tools.get_name_from_description(client, dummy_description)
    assert name == "Harold Wetherby"


def test_get_age_from_description():
    client = OpenAI()

    age = tools.get_age_from_description(client, dummy_description)
    assert age == "66"
