import os
import openai
import json
from dotenv import load_dotenv


def split_string(string: str, chunk_size: int):
    return [string[i : i + chunk_size] for i in range(0, len(string), chunk_size)]


class AIService:
    def __init__(self):
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        openai.api_key = OPENAI_API_KEY

    def generate_prompt(self, stats: dict):
        # fin_aid = (
        #     "interested in financial aid"
        #     if stats["fin_aid"]
        #     else "not interested in financial aid"
        # )
        prompt = (
            "I am a student who wants to study in the US. I am from "
            + "Kazakhstan"
            # + stats["country"]
            + ". I am interested in "
            + stats["majors"]
            + " major"
            + ". I have a GPA of "
            + str(stats["CGPA"])
            + " out of "
            + str(stats["GPA_scale"])
            + ". I have a SAT score of "
            + str(stats["sat_score"])
            + ". I have an IELTS score of "
            + str(stats["ielts_score"])
            # + ". I am "
            # + fin_aid
            # + ". I have studied at "
            # + stats["school"]
            # + ". My hobbies are: "
            # + stats["interests"]
            # + ". I have participated in "
            # + stats["olympiads"]
            # + ". I have worked on "
            # + stats["projects"]
            # + ". I have volunteered at "
            # + stats["volunteering"]
            + "."
            + "Please, provide a list of 10-12 universities for me in "
            + str(stats["country"])
            + ". Do not make up universities. Make sure to provide their actual names, "
            + "types (reach, target or safety), descriptions and tips on applying to them. Do not recommend universities as "
            + "'Reach University 1, Reach University 2, Target University 1' etc. Be strict in your judgements."
            + "You should keep in mind the following:  Understand, that in applicant's country, GPA of 4.50/5.00 is considered slightly above average. "
            + "and not enough for applying to top universities. "
            + "An applicant is eligible to apply to Ivy league, Harvard, MIT and other top universities only if their GPA is close to the maximum "
            + "(above 90 percent of the maximum), "
            + "they have an exceptional SAT score of more than 1400 and IELTS score of more than 7.0. If an applicant is weaker than an average applicant to"
            + " some reach university, do not recommend it, just skip it."
        )
        # print(prompt)
        return prompt

    def generate_response(self, prompt: str) -> list[str]:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful college assistant. Your job is to provide an INTERNATIONAL applicant with a list"
                    + " of 10-12 universities in their country of preferrence. With that in mind, give your recommendations."
                    + " Give each university's actual name and type (reach, target or safety)."
                    + " Then, provide a through description of each of the universities and its relevance to the applicant."
                    + " Include some interesting facts about the universities in their descriptions, if you can."
                    + " Finally, provide tips on applying to each university. Be strict in your judgements."
                    + " Before describing, include the university's type (reach, target or safety). Do not make up any information!"
                    + " Example of bad names: 'Reach University 1, Target University 1, Safety University 1' etc. "
                    + " Your answer should ONLY contain the university list."
                    # Style your answer as a JSON array of "
                    # + "Objects with the following fields: 'name' which contains the name of the university, 'type' which contains the type of the university "
                    # + "(reach target or safety), 'description' which should include  the reason you recommend this university to an applicant,"
                    # + " and 'tips' which should include your tips on applying to this university. "
                    # + "Example of bad names: 'Reach University 1, Target University 1, Safety University 1' etc. "
                    # + " Your answer should ONLY contain JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
        )
        reply = response.choices[0].message.content.strip()
        return reply

    def generate_unilist_json(self, universityList: str):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful college assistant. You will be given a text with a list of universities, "
                    + "their types (reach, target or safety), descriptions and tips on applying to them. "
                    + " Your job is to convert this text into a JSON array of Objects with the following fields: "
                    + "'name' which contains the name of the university, 'type' which contains the type of the university "
                    + "(reach target or safety), 'description' which should include its description"
                    + "and 'tips' which should include tips on applying to this university. "
                    + "Your answer should ONLY contain JSON.",
                },
                {
                    "role": "user",
                    "content": "Convert the following text into JSON" + universityList,
                },
            ],
            temperature=0.9,
        )
        reply = response.choices[0].message.content.strip()
        reply_json = json.loads(reply)
        return reply_json

    def generate_roadmap(self, university_name: str, stats: dict):
        prompt = (
            "I am a student who wants to study in "
            + stats["country"]
            + ". I am interested in "
            + stats["majors"]
            + " major"
            + ". I have a GPA of "
            + str(stats["CGPA"])
            + " out of "
            + str(stats["GPA_scale"])
            + ". I have a SAT score of "
            + str(stats["sat_score"])
            + ". I have an IELTS score of "
            + str(stats["ielts_score"])
            + ". Please, provide a roadmap for an applicant with my stats "
            + "on applying to "
            + university_name
            + ". Do not make up any information! Style your roadmap as a list of steps."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful college assistant. Your job is to "
                    + "provide an INTERNATIONAL applicant with a roadmap on applying to"
                    + " a university. Keep in mind that an applicant is in need of "
                    + "financial aid. Do not hesitate to critique the applicant's stats"
                    + " and give them advice on how to improve their application."
                    + ". With that in mind, give your recommendations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
        )
        reply = response.choices[0].message.content.strip()
        return reply

    def generate_roadmap_json(self, roadmap: str):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful college assistant. You will be given a text with a roadmap on applying to a university. "
                    + " Your job is to convert this text into a JSON with the following fields: "
                    + "'introduction' which contains the introduction to the roadmap, 'conclusion' which contains the conclusion, and "
                    + "'steps' which is an array that contains all of the steps (strictly in order, but without the step number). "
                    + "Your answer should ONLY contain JSON.",
                },
                {
                    "role": "user",
                    "content": "Convert the following text into JSON" + roadmap,
                },
            ],
            temperature=0.9,
        )
        reply = response.choices[0].message.content.strip()
        reply_json = json.loads(reply)
        return reply_json
