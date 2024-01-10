import os
from typing import List

import openai
import requests
from dotenv import load_dotenv

from backend.schemas.case import Case, CaseQuery

load_dotenv()


async def get_cases_by_name(
    case_name,
    citation,
) -> List[CaseQuery]:
    params = [
        (case_name, "&name_abbreviation="),
        (citation, "&cite="),
    ]
    url = "https://api.case.law/v1/cases/?"
    for p in params:
        if p[0]:
            url += f"{p[1]}{p[0]}"
    results_obj = requests.get(url)
    query_results = results_obj.json()

    cases = []

    results = query_results["results"]
    for r in results:
        case = {}
        case["case_name"] = r["name"]
        case["id"] = r["id"]
        case["citation"] = r["citations"][0]["cite"]
        cases.append(case)

    return cases


async def get_case_by_id(id, include_summary) -> Case:
    url = f"https://api.case.law/v1/cases/{id}"
    case_obj = requests.get(url)
    raw_case = case_obj.json()

    keys = [
        ("case_name", None, "name"),
        ("id", None, "id"),
        ("short_title", None, "name_abbreviation"),
        ("jurisdiction", "jurisdiction", "name"),
        ("jurisdiction_id", "jurisdiction", "id"),
        ("court", "court", "name"),
        ("court_id", "court", "id"),
        ("docket_number", None, "docket_number"),
        ("reporter_volume", "volume", "volume_number"),
        ("reporter", "reporter", "full_name"),
        ("reporter_id", "reporter", "id"),
        ("first_page", None, "first_page"),
        ("date_decided", None, "decision_date"),
        ("citation", "citations", "cite"),
        ("case_type", "citations", "type"),
        ("frontend_pdf_url", None, "frontend_pdf_url"),
    ]

    case = {}

    for key in keys:
        if key[1] is None:
            case[f"{key[0]}"] = raw_case[f"{key[2]}"]
        elif type([raw_case[f"{key[1]}"]][0]) is dict:
            case[f"{key[0]}"] = raw_case[key[1]][key[2]]
        else:
            case[f"{key[0]}"] = raw_case[key[1]][0][key[2]]

    if include_summary:
        case["summary"] = await get_summary(case_name=case["case_name"])
    else:
        case["summary"] = None

    return case


async def get_summary(case_name) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an informative assistant that provides an in-depth summary of the details of a court case including the majority opinion and the dissenting opinion.",  # noqa
            },
            {"role": "user", "content": case_name},
        ],
        temperature=0.7,
        max_tokens=2400,
    )

    message = response["choices"][0]["message"]["content"]

    return message


"""
async def create_ris_file(id: int):
    case = await get_case_by_id(id=id, include_summary=False)
    buffer = BytesIO()
    file = open("export.ris", "w")

    content = [
        "TY  - CASE \n",
        f"TI  - {case['case_name']} \n",
        f"A2  - {case['reporter']} \n",
        f"AB  - {case['summary']} \n",
        f"DA  - {case['date_decided']} \n",
        f"PY  - {(case['date_decided']).split('-')[0]} \n",
        f"VL  - {case['reporter_volume']} \n",
        f"SP  - {case['first_page']} \n",
        f"PB  - {case['court']} \n",
        f"SV  - {case['docket_number']} \n",
        f"UR  - {case['frontend_pdf_url']} \n",
        f"M1  - citation: {case['citation']}  \n",
        "ER  - ",
    ]

    file.writelines(content)

    file.close()

    return FileResponse("export.ris", media_type="application/x-research-info-systems")
"""
