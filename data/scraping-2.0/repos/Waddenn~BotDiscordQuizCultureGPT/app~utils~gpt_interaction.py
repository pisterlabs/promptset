import json
import openai
import re
import random
from app.utils.wikipedia_interaction import get_random_wikipedia_title, get_article_details

mod = "infini"
quiz_info = {}


def load_from_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_random_item(data_list: list) -> str:
    return random.choice(data_list)


themes = load_from_json("./config/themes.json")["themes"]


def create_prompt(mod: str = None, theme: str = None) -> str:
    if mod == "infini":
        theme = get_random_wikipedia_title()
        details = get_article_details(theme)
        description = details["description"]
        url = details["url"]
        url = details["url"]
        quiz_info["current_url"] = url

    else:
        theme = mod or get_random_item(themes)

    return (
        f"{description}"
        f"écris une question courte de culture générale sur le thème '{theme}' en suivant ce format : "
        "Question: [Votre question ici] "
        "1: [Option correcte] "
        "2: [Option 2] "
        "3: [Option 3] "
        "4: [Option 4] "
    )


def is_correct_format(response: str) -> bool:
    lines = response.split("\n")
    if len(lines) != 5 or not lines[0].startswith("Question:"):
        return False
    return all(line.startswith(f"{i}:") for i, line in enumerate(lines[1:], 1))


def clean_response_format(response: str) -> str:
    response = re.sub(r"Question\s*:", "Question:", response).strip()
    lines = response.split("\n")
    return "\n".join(lines[:5])


def format_api_response(response) -> str:
    formatted_response = " ".join(response.choices[0].message["content"].split())
    formatted_response = re.sub(r"(\d+:)", r"\n\1", formatted_response)
    formatted_response = re.sub(r"(Question: )", r"\n\1", formatted_response)
    return clean_response_format(formatted_response)


def get_api_response(prompt: str):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=150,
        temperature=0.3,
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0,
        messages=[
            {
                "role": "system",
                "content": "You are a structured-response generator. Provide a structured response based on solid verified facts",
            },
            {"role": "user", "content": prompt},
        ],
    )


def get_formatted_response(mod: str = None, theme: str = None) -> str:
    prompt = create_prompt(mod, theme)
    print(prompt)
    for _ in range(3):
        try:
            response = get_api_response(prompt)
            formatted_response = format_api_response(response)
            print(formatted_response)
            if is_correct_format(formatted_response):
                return formatted_response
        except Exception as e:
            return f"Erreur lors de l'interaction avec OpenAI: {e}"
    return "Erreur : Impossible d'obtenir une réponse au format correct après trois tentatives."
