import datetime

import requests
from django.conf import settings
from openai.error import ServiceUnavailableError

from .ai import chat

RECOMMENDATION_SPLIT_STRING = "\n\n      \n          \n          \n\n\n\n              \n                \n        \n              \n  \n\n      \n          "


LINKEDIN_API_EXPLAINATION = {
    "experiences": "the experiences",
    "education": "the educational institutions attended",
    "publications": "their publications",
    "honors_awards": "their awards",
    "patents": "their patents",
    # "recommendations": "the recommendations",
    "volunteer_work": "the volunteer work",
}

CHATGPT_REWORD_PROMPT = "I want to write a description for an activity I have. If any part of it sounds bad, then reword it without falsifying any info. Do not make it too long, and do not add an intro or ending:"
CHATGPT_SUMMARY_PROMPT = "I want to write a SHORT paragraph for my professional portfolio. Heres the content to use to write it (make it 3 sentences max, make the paragraph wholistic by focusing on the important skills, and don't falsify anything):"
CHATGPT_SKILLS_PROMPT = "Generate a bullet-point list of technical skills from this text (focus on technical skills that are applicable to industry, stuff like technical skills i have experience with):"


def _clean_str(s, empty=""):
    if s is None or s == "":
        return empty

    return s.replace("\r", "").strip()


def _clean_date(d):
    if d is None:
        return None

    return datetime.date(d["year"], d["month"], d["day"])


def _no_newline(s):
    return s.replace("\n", " ")


def fetch_linkedin_profile(linkedin_url):
    resp = requests.get(
        settings.PROXYCURL_API_ENDPOINT,
        params={"url": linkedin_url},
        headers={"Authorization": f"Bearer {settings.PROXYCURL_API_KEY}"},
    )
    resp.raise_for_status()

    data = resp.json()

    linkedin_data = {
        "header": {
            "linkedin_id": _clean_str(data["public_identifier"]),
            "profile_pic_url": _clean_str(data["profile_pic_url"], empty=None),
            "background_cover_url": _clean_str(
                data["background_cover_image_url"], empty=None
            ),
            "first_name": _clean_str(data["first_name"]),
            "last_name": _clean_str(data["last_name"]),
            "occupation": _clean_str(data["occupation"]),
            "headline": _clean_str(data["headline"]),
            "summary": _clean_str(data["summary"]),
            "country": _clean_str(data["country"]),
            "city": _clean_str(data["city"]),
            "state": _clean_str(data["state"]),
            "gender": _clean_str(data["gender"]),
            "birth_date": _clean_date(data["birth_date"]),
            "birth_date": _clean_date(data["birth_date"]),
            "industry": _clean_str(data["industry"]),
        },
        "experiences": [
            {
                "starts_at": _clean_date(experience["starts_at"]),
                "ends_at": _clean_date(experience["ends_at"]),
                "organization": _clean_str(experience["company"]),
                "url": _clean_str(experience["company_linkedin_profile_url"]),
                "title": _clean_str(experience["title"]),
                "description": _clean_str(experience["description"]),
                "location": _clean_str(experience["location"]),
                "logo_url": _clean_str(experience["logo_url"], empty=None),
            }
            for experience in data["experiences"]
        ],
        "education": [
            {
                "starts_at": _clean_date(experience["starts_at"]),
                "ends_at": _clean_date(experience["ends_at"]),
                "field": _clean_str(experience["field_of_study"]),
                "title": _clean_str(experience["degree_name"]),
                "organization": _clean_str(experience["school"]),
                "url": _clean_str(experience["school_linkedin_profile_url"]),
                "description": _clean_str(experience["description"]),
                "logo_url": _clean_str(experience["logo_url"], empty=None),
                "grade": _clean_str(experience["grade"]),
                "activities_societies": _clean_str(
                    experience["activities_and_societies"]
                ),
            }
            for experience in data["education"]
        ],
        "languages": [{"title": language} for language in data["languages"]],
        "organizations": [
            {
                "starts_at": _clean_date(organization["starts_at"]),
                "ends_at": _clean_date(organization["ends_at"]),
                "organization": _clean_str(organization["org_name"]),
                "title": _clean_str(organization["title"]),
                "description": _clean_str(organization["description"]),
            }
            for organization in data["accomplishment_organisations"]
        ],
        "publications": [
            {
                "title": _clean_str(publication["name"]),
                "organization": _clean_str(publication["publisher"]),
                "date": _clean_date(publication["published_on"]),
                "description": _clean_str(publication["description"]),
                "url": _clean_str(publication["url"]),
            }
            for publication in data["accomplishment_publications"]
        ],
        "honors_awards": [
            {
                "title": _clean_str(honour_award["title"]),
                "organization": _clean_str(honour_award["issuer"]),
                "date": _clean_date(honour_award["issued_on"]),
                "description": _clean_str(honour_award["description"]),
            }
            for honour_award in data["accomplishment_honors_awards"]
        ],
        "patents": [
            {
                "title": _clean_str(patent["title"]),
                "organization": _clean_str(patent["issuer"]),
                "date": _clean_date(patent["issued_on"]),
                "description": _clean_str(patent["description"]),
                "application_number": _clean_str(patent["application_number"]),
                "number": _clean_str(patent["patent_number"]),
                "url": _clean_str(patent["url"]),
            }
            for patent in data["accomplishment_patents"]
        ],
        "courses": [
            {
                "title": _clean_str(course["name"]),
                "number": _clean_str(course["number"]),
            }
            for course in data["accomplishment_courses"]
        ],
        "projects": [
            {
                "starts_at": _clean_date(project["starts_at"]),
                "ends_at": _clean_date(project["ends_at"]),
                "title": _clean_str(project["title"]),
                "description": _clean_str(project["description"]),
                "url": _clean_str(project["url"]),
            }
            for project in data["accomplishment_projects"]
        ],
        "test_scores": [
            {
                "title": _clean_str(test_score["name"]),
                "number": _clean_str(test_score["score"]),
                "date": _clean_date(test_score["date_on"]),
                "description": _clean_str(test_score["description"]),
            }
            for test_score in data["accomplishment_test_scores"]
        ],
        "volunteer_work": [
            {
                "starts_at": _clean_date(volunteer_work["starts_at"]),
                "ends_at": _clean_date(volunteer_work["ends_at"]),
                "title": _clean_str(volunteer_work["title"]),
                "cause": _clean_str(volunteer_work["cause"]),
                "organization": _clean_str(volunteer_work["company"]),
                "url": _clean_str(volunteer_work["company_linkedin_profile_url"]),
                "description": _clean_str(volunteer_work["description"]),
                "logo_url": _clean_str(volunteer_work["logo_url"], empty=None),
            }
            for volunteer_work in data["volunteer_work"]
        ],
        "certifications": [
            {
                "starts_at": _clean_date(certification["starts_at"]),
                "ends_at": _clean_date(certification["ends_at"]),
                "title": _clean_str(certification["name"]),
                "number": _clean_str(certification["license_number"]),
                # "display_source": _clean_str(certification["display_source"]),
                "organization": _clean_str(certification["authority"]),
                "url": _clean_str(certification["url"]),
            }
            for certification in data["certifications"]
        ],
        # "recommendations": [
        #     {
        #         "author": recommendation.split(RECOMMENDATION_SPLIT_STRING)[0],
        #         "description": recommendation.split(RECOMMENDATION_SPLIT_STRING)[1],
        #     }
        #     for recommendation in data["recommendations"]
        # ],
        "articles": [
            {
                "title": _clean_str(article["title"]),
                "url": _clean_str(article["url"]),
                "date": _clean_date(article["published_date"]),
                "author": _clean_str(article["author"]),
                "image_url": _clean_str(article["image_url"], empty=None),
            }
            for article in data["articles"]
        ],
        "interests": [{"title": interest} for interest in data["interests"]],
        "skills": [],
    }

    if settings.OPENAI_API_KEY is None:
        return linkedin_data

    chatgpt_context = ""
    for item_type, explaination in LINKEDIN_API_EXPLAINATION.items():
        if len(linkedin_data[item_type]) > 0:
            chatgpt_context += f"Here are all {explaination}:\n"
            for item in linkedin_data[item_type]:
                if item_type == "recommendations":
                    chatgpt_context += f'- {_no_newline(item["author"])}: {_no_newline(item["description"])}\n'
                else:
                    chatgpt_context += f'- {_no_newline(item["title"])}: {_no_newline(item["description"])}\n'

    try:
        if data["summary"] is None:
            linkedin_data["header"]["summary"] = _clean_str(
                chat(f"{CHATGPT_SUMMARY_PROMPT}\n\n{chatgpt_context}")
            )

        chatgpt_skills_response = chat(f"{CHATGPT_SKILLS_PROMPT}\n\n{chatgpt_context}")
        if "- " in chatgpt_skills_response:
            skills = (
                chatgpt_skills_response
                .replace("\n", "")
                .strip("- ")
                .split("- ")
            )
            linkedin_data["skills"] = [{"title": _clean_str(skill)} for skill in skills]
    except ServiceUnavailableError:
        pass
    return linkedin_data
