import datetime

import html2text
import markdown
import openai
from bs4 import BeautifulSoup

from app import babotree_utils
from app.database import get_direct_db
from app.models import HighlightSourceOutline, Highlight, HighlightSource


def find_section_boundaries(md_text):
    # Convert Markdown to HTML
    html = markdown.markdown(md_text)

    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Find all headers (h1, h2, h3, etc.)
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    print(headers)

    sections = []
    for i, header in enumerate(headers):
        # Determine the level of the current header
        level = int(header.name[1])

        # Find the next header of the same or higher level
        next_header = None
        for next_elem in header.find_all_next(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            next_header = next_elem
            break

        # Get all content between current header and next header
        content = []
        for elem in header.next_siblings:
            if elem == next_header:
                break
            content.append(str(elem))

        sections.append((header, ''.join(content)))

    return sections


def get_last_outline_generation():
    """
    A function that queries the database table highlight_sources_outlines for the most recent update, via the created_at field/column.

    If the table is empty it returns None
    :return:
    """
    return datetime.datetime.utcnow() - datetime.timedelta(days=30)
    db = get_direct_db()
    most_recent_outline_generation = db.query(HighlightSourceOutline).order_by(HighlightSourceOutline.created_at.desc()).first()
    if most_recent_outline_generation:
        return most_recent_outline_generation.created_at
    else:
        return None

openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
)
def generate_outline(source, source_highlights):
    """
    A function that takes a source and a list of highlights and generates an outline for that source using an LLM
    :param source:
    :param source_highlights:
    :return:
    """
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant with an expert knowledge in how to create study outlines on various topics."
    },
        {
            "role": "user",
            "content": f"Here are some excerpts from a source text called \"{source.title}\" :\n" + "\n".join(
                [highlight.text for highlight in source_highlights])
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "Please create an outline in markdown syntax about the key concepts in the excepts. You can include related information not present in the excerpts if you think it would be helpful. Take a deep breath. Work through the process step by step."
        }]
    response = openai_client.chat.completions.create(
        model='mistralai/Mixtral-8x7B-Instruct-v0.1',
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(f"--- LLM OUTLINE FOR \"{source.title}\" ---")
    print(response.choices[0].message.content)
    total_cost = response.usage.total_tokens * ((.6) / (1_000_000))
    print(f"--- outline used this many tokens: {response.usage.total_tokens} (${total_cost}) ---")
    print("--- END LLM OUTLINE ---")
    return response.choices[0].message.content


def main():
    # we want to check which highlights were updated since the last time
    # we generated outlines
    last_outline_generation = get_last_outline_generation()
    if not last_outline_generation:
        print("No outlines have been generated yet, will do so for books with highlights in the last 30 days")
        last_outline_generation = datetime.datetime.utcnow() - datetime.timedelta(days=30)

    # get all highlights since last outline generation
    db = get_direct_db()
    recent_highlights = db.query(Highlight).filter(Highlight.created_at > last_outline_generation).all()
    print(f"Found {len(recent_highlights)} highlights since last outline generation")
    # get all the sources for those highlights
    recent_source_ids = set([highlight.source_id for highlight in recent_highlights])
    print(f"Found {len(recent_source_ids)} sources for those highlights")
    for source_id in recent_source_ids:
        print(f"Regenerating outline for source {source_id}")
        # get all the highlights for that source
        source = db.query(HighlightSource).filter(HighlightSource.id == source_id).first()
        source_highlights = db.query(Highlight).filter(Highlight.source_id == source_id).all()
        # generate an outline for those highlights
        outline_md = generate_outline(source, source_highlights)
        # delete existing outline
        db.query(HighlightSourceOutline).filter(HighlightSourceOutline.source_id == source_id).delete()
        # save the new outline to the database
        outline_model = HighlightSourceOutline(source_id=source_id, outline_md=outline_md)
        db.add(outline_model)
        db.commit()


if __name__ == '__main__':
    main()