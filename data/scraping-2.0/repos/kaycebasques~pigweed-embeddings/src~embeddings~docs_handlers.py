import os

import bs4
import openai

import database
import utilities

def preprocess(mgr, url):
    old_text = mgr.get_text(url)
    soup = bs4.BeautifulSoup(old_text, 'html.parser')
    main = soup.select('div.main')[0]
    for tag in ['script', 'style', 'link']:
        for useless_node in main.find_all(tag):
            useless_node.decompose()
    new_text = str(main)
    mgr.set_text(url, new_text)

def segment(mgr, url):
    text = mgr.get_text(url)
    soup = bs4.BeautifulSoup(text, 'html.parser')
    for section in soup.find_all('section'):
        segment_text = str(section)
        section_id = section.get('id')
        if not section_id:
            continue
        segment_url = f'{url}#{section_id}'
        mgr.set_segment(segment_url, segment_text)

def embed(mgr, url, text, checksums):
    openai_client = openai.OpenAI(api_key=os.environ.get('OPENAI_KEY'))
    db = database.Database()
    if utilities.token_count(text) > utilities.max_token_count():
        return
    if db.row_exists(content=text, checksums=checksums):
        db.update_timestamp(content=text)
    else:
        embedding = openai_client.embeddings.create(
            input=text,
            model=utilities.embedding_model()
        ).data[0].embedding
        db.add(content=text, content_type='web', url=url, embedding=embedding)
