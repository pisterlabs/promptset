# Working example using guardrails based on code here:
# https://docs.guardrailsai.com/guardrails_ai/getting_started/

from django.conf import settings
from pydantic import BaseModel, Field
from typing import List, Optional
from guardrails.validators import ValidRange, ValidChoices
from rich import print
import guardrails as gd
import openai
#import os
from decouple import config

# doctors_notes = """Chest pain that may feel like pressure, tightness, pain, squeezing or aching.
#     &amp;Pain or discomfort that spreads to the shoulder, arm, back, neck, jaw, teeth or sometimes the upper belly.
#     &amp;Cold sweat.
#     &amp;Fatigue.
#     &amp;Heartburn or indigestion.
#     &amp;Lightheadedness or sudden dizziness.
#     &amp;Nausea.
#     &amp;Shortness of breath.
# """

doctors_notes = """49 y/o Male with chronic macular rash to face &amp; hair, worse in beard, eyebrows &amp; nares.
Itchy, flaky, slightly scaly. Moderate response to OTC steroid cream"""

prompt = """
${doctors_notes}

${gr.complete_json_suffix_v2}
"""

class Symptom(BaseModel):
    symptom: str = Field(description="Symptom that a patient is experiencing")
    affected_area: str = Field(description="What part of the body the symptom is affecting", validators=[ValidChoices(choices=['head', 'neck', 'chest'], on_fail="reask")])

class Medication(BaseModel):
    medication: str = Field(description="Name of the medication the patient is taking")
    response: str = Field(description="How the patient is responding to the medication")


class PatientInfo(BaseModel):
    gender: str = Field(description="Patient's gender")
    age: int = Field(validators=[ValidRange(min=0, max=100, on_fail="fix")])
    symptoms: List[Symptom] = Field(description="Symptoms that the patient is currently experiencing. Each symptom should be classified into a separate item in the list.")
    current_meds: List[Medication] = Field(description="Medications the patient is currently taking and their response")


# From pydantic:
guard = gd.Guard.from_pydantic(output_class=PatientInfo, prompt=prompt)
OAI_KEY = config('OPEN_AI_API_KEY')
# Wrap the OpenAI API call with the `guard` object
openai.api_key = OAI_KEY
raw_llm_output, validated_output = guard(
    openai.Completion.create,
    prompt_params={"doctors_notes": doctors_notes},
    engine="text-davinci-003",
    max_tokens=1024,
    temperature=0.3,
)

# Print the validated output from the LLM
print(validated_output)
print(raw_llm_output)

# print(guard.state.most_recent_call.history[0].rich_group)
# print(guard.state.most_recent_call.history[1].rich_group)
# print(guard.state.most_recent_call.tree)




from pydantic import BaseModel, Field
from typing import List, Optional
from guardrails.validators import ValidRange, ValidChoices
from rich import print
import guardrails as gd
import openai
#import os
from decouple import config


# results = """
# 1. "Super Size Me" (2004)
# Genre: Documentary
# Summary: Filmmaker Morgan Spurlock investigates the impact of fast food by eating only McDonald's food for 30 days. This eye-opening documentary explores the consequences of an unhealthy diet and the importance of making healthier choices.
#
# 2. "Fat, Sick & Nearly Dead" (2010)
# Genre: Documentary
# Summary: This inspiring documentary follows Joe Cross, who embarks on a 60-day juice fast to regain his health and lose weight. Along the way, he encounters others who have also transformed their lives with juicing.
#
# 3. "Julie & Julia" (2009)
# Genre: Comedy, Drama
# Summary: Based on true events, the movie tells the story of Julie Powell, who, discontented with her job, embarks on a challenge to cook all 524 recipes in Julia Child's cookbook, "Mastering the Art of French Cooking." This delightful film combines cooking, self-discovery, and a sprinkle of comedy.
#
# 4. "Hungry for Change" (2012)
# Genre: Documentary
# Summary: This informative documentary exposes the secrets of the diet and weight loss industry, shedding light on the tactics used to keep people trapped in a cycle of unhealthy eating. It shares empowering strategies for a healthier lifestyle.
# """
results = """
Assistant: Sure! I have a great romantic movie recommendation for you. Have you watched "The Notebook" (2004)? It's a beautiful and heartfelt romantic film.

Here are the details:
Title: The Notebook (2004)
Genre: Romance, Drama
Summary: Set in the 1940s, "The Notebook" tells the story of Noah Calhoun and Allie Hamilton, two young lovers from different social backgrounds who fall deeply in love. Despite their families' disapproval, they embark on a passionate summer romance. However, circumstances separate them, and years later, they reunite when Allie is engaged to another man. As they confront their past and the choices they've made, they must decide if their love can withstand the test of time.

I hope you enjoy watching "The Notebook"! Let me know if you need any more suggestions.
User: please suggest me family movie
Assistant: Absolutely! How about "Finding Nemo" (2003)? It's a heartwarming family movie that both kids and adults can enjoy together.

Here are the details:
Title: Finding Nemo (2003)
Genre: Animation, Adventure, Comedy
Summary: "Finding Nemo" follows the story of Marlin, a clownfish, who embarks on a daring journey across the ocean to find his son, Nemo, who has been captured by a scuba diver and placed in a fish tank. With the help of an unforgettable cast of characters, including the cheerful and forgetful Dory, Marlin overcomes numerous obstacles and encounters vibrant marine life as they search for Nemo.

"Finding Nemo" is visually stunning and filled with humor and heartwarming moments. It's a perfect choice for a family movie night! Let me know if you have any other preferences or if there's anything else I can assist you with.

"""
#
# results = {
#     "movies": [
#         {
#             "title": "Super Size Me",
#             "year": 2004,
#             "genre": "Documentary",
#             "summary": "Filmmaker Morgan Spurlock investigates the impact of fast food by eating only McDonald's food for 30 days. This eye-opening documentary explores
# the consequences of an unhealthy diet and the importance of making healthier choices."
#         },
#         {
#             "title": "Fat, Sick & Nearly Dead",
#             "year": 2010,
#             "genre": "Documentary",
#             "summary": "This inspiring documentary follows Joe Cross, who embarks on a 60-day juice fast to regain his health and lose weight. Along the way, he
# encounters others who have also transformed their lives with juicing."
#         },
#         {
#             "title": "Julie & Julia",
#             "year": 2009,
#             "genre": "Comedy, Drama",
#             "summary": "Based on true events, the movie tells the story of Julie Powell, who, discontented with her job, embarks on a challenge to cook all 524 recipes
# in Julia Child's cookbook, \"Mastering the Art of French Cooking.\" This delightful film combines cooking, self-discovery, and a sprinkle of comedy."
#         },
#         {
#             "title": "Hungry for Change",
#             "year": 2012,
#             "genre": "Documentary",
#             "summary": "This informative documentary exposes the secrets of the diet and weight loss industry, shedding light on the tactics used to keep people trapped
# in a cycle of unhealthy eating. It shares empowering strategies for a healthier lifestyle."
#         }
#     ]
# }

prompt = """

${results}

${gr.complete_json_suffix_v2}
"""


class Movie(BaseModel):
    title: str = Field(description="Name of the movie")
    year: int = Field(validators=[ValidRange(min=0, max=3000, on_fail="fix")])
    genre: str = Field(description="Genre of movie")
    summary: str = Field(description="Summary of movie")


class MovieInfo(BaseModel):
    movies: List[Movie] = Field(description="Each movie should be classified into a separate item in the list.")


# From pydantic:
guard = gd.Guard.from_pydantic(output_class=MovieInfo, prompt=prompt)
OPEN_AI_API_KEY=settings.OPEN_AI_KEY

# OAI_KEY = config('OPEN_AI_API_KEY')
OAI_KEY = OPEN_AI_API_KEY

# Wrap the OpenAI API call with the `guard` object
openai.api_key = OAI_KEY
raw_llm_output, validated_output = guard(
    openai.Completion.create,
    prompt_params={"results": results},
    engine="text-davinci-003",
    max_tokens=2048,
    temperature=0.3,
)

# Print the validated output from the LLM
print(validated_output)
print(raw_llm_output)

