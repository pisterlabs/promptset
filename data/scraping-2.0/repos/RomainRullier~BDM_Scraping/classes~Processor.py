import openai
from bs4 import BeautifulSoup
import requests
import json

class TextProcessor():
    def __init__(self, key):

        openai.api_key = key

    def prompt(self, text):

        if text.startswith('/translate'):
            return self.openai_translate(
                text.replace('/translate', '').strip()
            )
        elif text.startswith('/sumary'):
            return self.openai_text_sumary(
                text.replace('/sumary', '').strip()
            )
        elif text.startswith('/imagine'):
            return self.openai_image(
                text.replace('/imagine', '').strip()
            )
        elif text.startswith('/code'):
            return self.openai_code(
                text.replace('/code', '').strip()
            )
        elif text.startswith('/generate'):
            return self.openai_text_generator(
                text.replace('/generate', '').strip()
            )
        elif text.startswith('/actu'):
            return self.summary_actu(
                text.replace('/actu', '').strip()
            )

        elif text.startswith('/json'):
            return self.json_from_20_minutes(
                text.replace('/json', '').strip()
            )

        return {
            'type': 'text',
            'content' : 'Je ne comprends pas votre demande'
        }

    def openai_translate(self, msg):

        return_gpt = openai.Completion.create(
            model="text-davinci-003",
            prompt=msg,
            temperature=0.3,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        return {
            'type': 'text',
            'content' : return_gpt['choices'][0]['text']
        }

    def openai_text_sumary(self, msg):

        return_gpt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": 'Ton ojectif est de synthétiser un texte.'},
                {"role": "user",
                 "content": "Je veux résumer le texte suivant : %s" % msg},
            ],
            max_tokens=100,
            temperature=0.9,
        )

        return {
            'type': 'text',
            'content' : return_gpt['choices'][0]['message']['content']
        }

    def openai_text_generator(self, msg):

        return_gpt = openai.Completion.create(
            model="davinci",
            prompt=msg,
            temperature=0.3,
            max_tokens=100
        )

        return {
            'type': 'text',
            'content' : return_gpt['choices'][0]['text']
        }

    def openai_code(self, msg):

        return_gpt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                    "content": 'Ton ojectif est de vérifier la syntaxe d\'un code.'},
                {"role": "user",
                    "content": "Je veux vérifier le code suivant : ```%s```" % msg},
            ],
            temperature=0.3,
            max_tokens=100
        )

        return {
            'type': 'code',
            'content' : return_gpt['choices'][0]['message']['content']
        }



    def openai_image(self, msg):

        return_gpt = openai.Image.create(
            prompt=msg,
            n=1,
            size="512x512"
        )

        return {
            'type': 'image',
            'content': return_gpt['data'][0]['url']
        }

    def summary_actu(self, msg):

        text = requests.get(f'https://www.bing.com/news/search?q={msg}').text
        soup = BeautifulSoup(text, 'html.parser')

        actu = ' '.join(["- Actualité : " + link.text + ' \n' for link in soup.find_all('a', 'title')])

        return_gpt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"Tu es un rédacteur web qui synthétise l'actualité en 50 mots sur la thématique '{msg}' Tu fais des liaisons entre les articles avec des mots tel que 'mais', 'donc', 'or', 'par contre', 'en revanche', 'en effet', 'cependant', 'toutefois', 'par ailleurs', 'par contre', 'par contre, 'enfin'"},
                {"role": "user",
                 "content": "Voici la liste des actualités à synthétiser :" + actu},
            ],
            max_tokens=100,
            temperature=0.9,
        )

        return {
            'type': 'text',
            'content' : return_gpt['choices'][0]['message']['content']
        }

    def json_from_20_minutes(self, msg):

        response = requests.get(msg).text
        soup = BeautifulSoup(response, "html.parser")

        text = ' '.join([article.text + (article.get('href') if article.get('href')  else '') + ((article.get('src') if article.get('src')  else '')) for article in soup.find_all('article')])

        # ask openai to create a json
        return_gpt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": 'Ton ojectif est de créer un json à partir d\'un texte. Tu dois retourner uniquement le '
                            'json. Le json doit etre valide tu dois le parser pour être utiliser par python, '
                            'dans le json je veux le titre, la catégorie, l"image, le sous titre, le lien vers l"article etc..., '
                            'je veux uniquement le json rien d"autres avec pas de voici le json par exemple'},
                {"role": "user",
                 "content": "Je veux créer un json à partir du texte suivant : %s" % text[:4000]},
            ],
            temperature=0.3,
        )

        return {
            'type': 'json',
            'content' : return_gpt['choices'][0]['message']['content']
        }

    def save_message(self, user_msg, gpt_msg):
        data = {
            'user_msg': user_msg,
            'gpt_msg': gpt_msg
        }
        with open('data.json', 'a') as outfile:
            json.dump(data, outfile)
