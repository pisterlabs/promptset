from itemadapter import ItemAdapter
import openai
from dotenv import load_dotenv
import os
load_dotenv()

openai.api_key = os.getenv("API_KEY")


class WebcrawlerPipeline:
    def process_item(self, item, spider):
        return item


import sqlalchemy as db

class DataBase():
    def __init__(self, name_database='database'):
        self.name = name_database
        self.url = f"sqlite:///{name_database}.db"
        self.engine = db.create_engine(self.url)
        self.connection = self.engine.connect()
        self.metadata = db.MetaData()
        self.table = self.engine.table_names()


    def create_table(self, name_table, **kwargs):
        colums = [db.Column(k, v, primary_key = True) if 'id_' in k else db.Column(k, v) for k,v in kwargs.items()]
        db.Table(name_table, self.metadata, *colums)
        self.metadata.create_all(self.engine)
        print(f"Table : '{name_table}' are created succesfully")

    def read_table(self, name_table, return_keys=False):
        table = db.Table(name_table, self.metadata, autoload=True, autoload_with=self.engine)
        if return_keys:table.columns.keys()
        else : return table


    def add_row(self, name_table, **kwarrgs):
        name_table = self.read_table(name_table)

        stmt = (
            db.insert(name_table).
            values(kwarrgs)
        )
        self.connection.execute(stmt)
        print(f'Row id added')


    def delete_row_by_id(self, table, id_):
        name_table = self.read_table(name_table)

        stmt = (
            db.delete(name_table).
            where(students.c.id_ == id_)
            )
        self.connection.execute(stmt)
        print(f'Row id {id_} deleted')

    def select_table(self, name_table):
        name_table = self.read_table(name_table)
        stm = db.select([name_table])
        return self.connection.execute(stm).fetchall()
    

import openai
import bs4
import requests


class TextProcessor:
    def __init__(self):
        pass
    
    def openai_translate(self,texte, language):
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Translate the text after in {language}:\n\n{texte}\n\n1.",
        temperature=0.3,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
        response.choices[0].text
        return response.choices[0].text
    
    def openai_text_summary(self,texte):
        reponse = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                "content": f"resume moi le texte suivant: '{texte}' Tu fais des liaisons entre les articles avec des mots tel que 'mais', 'donc', 'or', 'par contre', 'en revanche', 'en effet', 'cependant', 'toutefois', 'par ailleurs', 'par contre', 'par contre, 'enfin'"},
                {"role": "user",
                "content": "Voici la liste des actualités à synthétiser :" + texte},
            ],
            max_tokens=100,
            temperature=0.9,
        )

        return reponse['choices'][0]['message']["content"]
    
    def openai_text_generator(self,texte):

        text = requests.get(f'https://www.bing.com/news/search?q={texte}').text
        soup = bs4.BeautifulSoup(text, 'html.parser')

        actu = ' '.join(["- Actualité : " + link.text+ ' \n'  for link in soup.find_all('a', 'title')])
        actu[:5]
        actu.split('\n')
        reponse = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                "content": f"Tu es un rédacteur web qui synthétise l'actualité en 50 mots sur la thématique '{texte}' Tu fais des liaisons entre les articles avec des mots tel que 'mais', 'donc', 'or', 'par contre', 'en revanche', 'en effet', 'cependant', 'toutefois', 'par ailleurs', 'par contre', 'par contre, 'enfin'"},
                {"role": "user",
                "content": "Voici la liste des actualités à synthétiser :" + actu},
            ],
            max_tokens=100,
            temperature=0.9,
        )

        return reponse


    def openai_codex(self,texte):
        reponse = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "assistant",
            "content": f"Corrige le code envoyé"},
            {"role": "user",
            "content": "Voici le code à corriger :" + texte},
        ],
        max_tokens=200,
        temperature=0.9,
    )

        return reponse

    def openai_image(self,texte):
        response = openai.Image.create(
        prompt=texte,
        n=1,
        size="1024x1024",
        )
        image_url = response['data'][0]['url']
        image_url

        # response = openai.Image.create_variation(
        # image=image_url,
        # n=1,
        # size="1024x1024"
        # )
        # image_url = response['data'][0]['url']

        return image_url
    



    def test(self, text):
        self.text = text
        print(self.text)
        return f"Texte traité : {self.text}"
