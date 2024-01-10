import os

import openai

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv("OPENAI_ENDPOINT")
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_openai_sympthoms(prompt: str):
    response = openai.Completion.create(
        engine="TextDaVinci003",
        prompt=f"""
        Podsumuj objawy pacjenta na podstawie transkrypcji rozmowy w pojedyńczych prostych zdaniach/słowach.
        Przykład: Ból gardła, rano. Brak gorączki. Problemy ze snem od dłuższego czasu.
        Rozmowa: {prompt}
        """,
        max_tokens=750,
        temperature=0,
    )
    return response['choices'][0]['text']

def get_openai_recommendations(prompt: str):
    response = openai.Completion.create(
        engine="TextDaVinci003",
        prompt=f"""
        Podsumuj zalecenia lekarza oraz jakie leki powinien przyjmować pacjent na podstawie rozmowy. Nie dodawaj nic od siebie! Napisz bardzo zwięźle.
        Rozmowa: {prompt}
        """,
        max_tokens=750,
        temperature=0,
    )
    return response['choices'][0]['text']