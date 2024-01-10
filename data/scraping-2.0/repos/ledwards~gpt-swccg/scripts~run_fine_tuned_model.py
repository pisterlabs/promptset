import os
from dotenv import load_dotenv
import openai

load_dotenv()

################################################################################
# NOTE: This doesn't work, as Answer API doesn't support fine-tuned models yet #
################################################################################

openai.organization = "org-D2FBgBhwLFkKAOsgtSp86b4i"
openai.api_key = os.getenv("OPENAI_API_KEY")

remote_files = openai.File.list()["data"]
training_files = filter(lambda f: "training.jsonl" in f["filename"], remote_files)
latest_file_id = max(training_files, key=lambda x: x["created_at"])["id"]

fine_tunes = openai.FineTune.list()["data"]
latest_fine_tuned_model_id = max(fine_tunes, key=lambda x: x["created_at"])["id"]

questions = [
    "What destiny is Wedge Antilles?",
    "What card cancels Cloud City Celebration?",
    "What gender is Toryn Farr?",
    "What race is Bail Organa?",
    "What planet is Mon Mothma from?",
    "Which Effect downloads Luke's Lightsaber?",
    "Which Dark Jedi Master uploads Force Lightning?",
    "Which Objective deploys Naboo: Swamp?",
    "How many lightsabers can General Greivous use?",
    "Which Dark Jedi Master is destiny 6?",
    "How many Light Side Jabba's Palace sites are there?"
]

for question in questions:
    answer = openai.Answer.create(
        search_model="ada",
        model=latest_fine_tuned_model_id,
        question=question,
        examples_context="Captain Jean-Luc Picard is a Light Side character card. Captain Jean-Luc Picard has a power of 5.",
        examples=[["What Power is Jean-Luc Picard?", "Captain Jean Luc Picard is Power 5"]], 
        file=latest_file_id,
        max_rerank=50,
        max_tokens=20,
        stop=["\n", "<|endoftext|>"]
    )
    print(question)
    print(f'> {answer["answers"][0]}')
    