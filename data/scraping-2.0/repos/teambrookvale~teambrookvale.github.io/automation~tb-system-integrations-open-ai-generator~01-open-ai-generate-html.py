import sqlite3
import itertools
import openai
import json
import re
import time
import os

ROOT_FOLDER = 'automation/tb-system-integrations-open-ai-generator'
HTML_FOLDER = f'{ROOT_FOLDER}/html'
JSON_FOLDER = f'{ROOT_FOLDER}/json'
MD_FOLDER = f'collections/_landing_system_integrations_articles'

md_files = os.listdir(MD_FOLDER)
md_file_names = [os.path.splitext(x)[0] for x in md_files]

html_files = os.listdir(HTML_FOLDER)
html_file_names = [os.path.splitext(x)[0] for x in html_files]

#count intesection of md_files and html_files
print(f'{len(set(md_file_names).intersection(html_file_names))} of {len(html_file_names)} md files already generated')


conn = sqlite3.connect('open-ai-generator-sqlite.db')
conn.execute('CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY, platform_1 TEXT, platform_2 TEXT, file_name TEXT, html TEXT, response TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)')

openai.api_key = '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'

def generate_open_ai_messages(platfrom_1, platform_2):
    prompt = f'''
    Write blog post in HTML, make sure to include <title> tag and conclusion section and cover the following:
    - {platfrom_1}
    - {platform_2}   
    - Integration of the two through API or SDK
    - Problems their integration solves
    - Conclusion
    '''
    messages=[{"role": "user", "content":prompt}]

    return messages

def is_generated(platform_1, platform_2):
    s12 = conn.execute(f"SELECT COUNT(*) FROM posts WHERE platform_1 = '{platform_1}' AND platform_2 = '{platform_2}'").fetchall()[0][0] > 0
    #s21 = conn.execute(f"SELECT COUNT(*) FROM posts WHERE platform_1 = '{platform_2}' AND platform_2 = '{platform_1}'").fetchall()[0][0] > 0
    return s12 #or s21

def insert_blog_post(platform_1, platform_2, file_name, html, response):
    conn.execute("INSERT INTO posts (platform_1, platform_2, file_name, html, response) values (?, ?, ?, ?, ?)", (platform_1, platform_2, file_name, html, response))
    conn.commit()

def url_safe(s):
    return re.sub(r'\W+', '-', s).lower()

def url_safe_platform_combination(pc):
    return pc[0] + '-' + pc[1]

platforms = []

with open(f'{ROOT_FOLDER}/zapier-premier-platforms.txt', 'r') as file:
    platforms = [x.strip() for x in file.readlines()]

platform_combinations = list(itertools.product(platforms, platforms))

for p in platform_combinations:
    #if p[0] == p[1] or is_generated(p[0], p[1]):
    #    continue

    file_name = url_safe_platform_combination(p)

    if file_name in md_file_names:
        continue

    print(f'INFO: Generating {p[0]} vs {p[1]}')

    messages = generate_open_ai_messages(p[0], p[1])

    while True:
        try:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

            with open(f"{JSON_FOLDER}/{file_name}.json", "w", encoding='utf-8') as file:
                file.write(json.dumps(response))

            with open(f"{HTML_FOLDER}/{file_name}.html", "w", encoding='utf-8') as file:
                file.write(response.choices[0].message.content)

            insert_blog_post(p[0], p[1], file_name, response.choices[0].message.content, json.dumps(response))

            break
        
        except Exception as e:
            if hasattr(e, 'message'):
                print(f"ERROR: {e.message}")
            else:
                print(f"ERROR: {e}")

            time.sleep(120)

