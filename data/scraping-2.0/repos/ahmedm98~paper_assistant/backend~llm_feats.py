import json
import os

from dotenv import load_dotenv
from extract_text import extract_text_from_xml
from grobid_client_python.grobid_client.grobid_client import GrobidClient
from openai import OpenAI


def process_pdf_grobid(
    file, config_path="configs/grobid_config.json", output="./files/"
):
    client = GrobidClient(config_path=config_path)
    client.process(
        "processFulltextDocument", "files/", output=output, force=True
    )


def send_prompt_to_openai(prompt, system_role):
    load_dotenv()

    with open("configs/openai_config.json", "r") as jsonfile:
        config = json.load(jsonfile)

    api_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ],
    )
    return response


def get_summary(paper: str):
    print(paper)
    process_pdf_grobid(file=f"files/{paper}")
    file_location = f"files/{paper.replace('.pdf','')}.grobid.tei.xml"
    print(file_location)
    if os.path.exists(file_location):
        full_text = extract_text_from_xml(file_location)
    else:
        print("The xml file does not exist")
        return None

    # noqa: E501
    prompt = f"""Summarize the text below. The summary should be 300 characters max and describes what this paper is about.

    {full_text}
    """  # noqa: E501

    system_role = "You are a helpful summarizer."
    response = send_prompt_to_openai(prompt=prompt, system_role=system_role)
    summary = response.choices[0].message.content
    return summary


def get_embedding(text, model="text-embedding-ada-002"):
    load_dotenv()

    api_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=api_key)

    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=model).data[0].embedding
    )
