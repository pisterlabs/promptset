import openai
import os
from tinydb import TinyDB, Query
from utils.fileio import download_image
from .monster import MonsterCard


def monster_generator(parent1=None, parent2=None, amount: int = 1, path: str='/home/cameron/Projects/StickerMonsters/monster_project/'):
    '''Generates monsters and saves it to tinyDB and posts to Django app.'''
    monster_db = TinyDB('monster_db.json')

    monster_name = None
    monster_element = None
    if parent1 is not None and parent2 is not None:
        monster_name = f'{parent1.creature}-{parent2.creature}'
        monster_element = f'{parent1.element_type}-{parent2.element_type}'

    for i in range(amount):
        monster = make_monster(monster_name, monster_element, path)
        monster_db.insert(monster.__dict__())

    printed = Query()
    monsters = monster_db.search(printed.print_status == False)

    monster_ids = []
    for monster in monsters:
        monster_ids.append(monster['uuid'])
    return monster_ids

def make_monster(monster_name, monster_element, path: str, num=''):
    '''Generates a monster and returns it.'''
    
    creature = MonsterCard(monster_name, monster_element, path+'static/monsterimages')
    image_path = f'{creature.image_path}'.replace('.png', f'{num}.png')
    if os.path.exists(image_path):
        if num == '':
            num = 1
        else:
            num += 1
        return make_monster(monster_name, monster_element, path, num)
    try:
        creature.filename = f'{creature.filename}'.replace('.png', f'{num}.png')
        creature.image_path = image_path
        generate_monster(image_path, creature, 1)
    except Exception as e:
        print("Error generating monster, trying again.")
        print(e)
        return make_monster(monster_name, monster_element, path)
    
    creature.post_monster()
    qr_path = f"{path}monsterqrcodes/{creature.filename}"
    generate_qr_code(creature.url, qr_path)
    creature.qr_code_path = qr_path

    return creature

def generate_monster(path, creature, amount):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Image.create(
    prompt=creature.prompt,
    n=amount,
    size="512x512"
    )
    for image in response['data']:
        url = image['url']
        download_image(url, path)

def generate_qr_code(url, path):
    api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={url}"
    download_image(api_url, path)
