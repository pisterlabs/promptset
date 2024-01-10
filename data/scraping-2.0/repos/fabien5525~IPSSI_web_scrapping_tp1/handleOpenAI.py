import openai
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def commandList() -> list:
    return [
        '/translate',
        '/summary',
        '/imagine',
        '/code',
        '/actu',
        '/json'
    ]

def handleOpenAI(api_key, prompt) -> str:
    if not prompt.startswith('/') or prompt.split(' ')[0] not in commandList():
        return 'Commande non reconnue'  + '\n' + 'Liste des commandes : ' + '\n' + '\n'.join(commandList())

    prompt = prompt.strip()
    command = prompt.split(' ')[0]
    search = prompt.replace(command, '').strip()

    openai.api_key = api_key

    match command:
        case '/translate':
            return handleTranslate(search)
        case '/summary':
            return handleSummary(search)
        case '/imagine':
            return handleImagine(search)
        case '/code':
            return handleCode(search)
        case '/actu':
            return handleActu(search)
        case '/json':
            return handleJson(search)

    return ''

def handleTranslate(prompt) -> str:
    customPrompt = 'Translate to french if english or to english if french (with a response like "Traduction : ") : \n\n {}'.format(prompt)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=customPrompt,
        temperature=0.3,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    try: 
        return response.choices[0].text
    except:
        print('Une erreur est survenue avec la commande /translate')
        print(response)
        return 'Une erreur est survenue'

def handleSummary(prompt) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
            "content": f"Tu es un rédacteur web qui synthétise l'actualité en 50 mots, Tu fais des liaisons entre les articles avec des mots tel que 'mais', 'donc', 'or', 'par contre', 'en revanche', 'en effet', 'cependant', 'toutefois', 'par ailleurs', 'par contre', 'par contre, 'enfin'"},
            {"role": "user",
            "content": "Voici la liste des actualités à synthétiser : " + prompt},
        ],
        max_tokens=100,
        temperature=0.9,
    )

    try:
        return response.choices[0].message.content
    except:
        print('Une erreur est survenue avec la commande /summary')
        print(response)
        return 'Une erreur est survenue'

def handleImagine(prompt) -> str:
    customPrompt = prompt

    response = openai.Image.create(
        prompt=customPrompt,
        n=1,
        size="256x256"
    )

    try:
        return response['data'][0]['url']
    except:
        print('Une erreur est survenue avec la commande /imagine')
        print(response)
        return 'Une erreur est survenue'

def handleCode(prompt) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
            "content": "Tu es un expert informatique dans tous les langages, tu dois corriger le code ci dessous mais sans ajoiter de commentaire ou d'expliquation, tu dois juste corriger le code"},
            {"role": "user",
            "content": prompt},
        ],
        max_tokens=100,
        temperature=0.9,
    )

    try:
        return response.choices[0].message.content
    except:
        print('Une erreur est survenue avec la commande /code')
        print(response)
        return 'Une erreur est survenue'

def handleActu(prompt) -> str:
    response = requests.get("https://www.20minutes.fr/search?q={}#gsc.tab=0&gsc.q=IA&gsc.page=1".format(quote(prompt))).text
    soup = BeautifulSoup(response, "html.parser")
    text = soup.text.replace("\n", " ").replace("\t", " ")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
            "content": f"Tu es un rédacteur web qui synthétise l'actualité en une cinquentaine de mots, Tu fais des liaisons entre les articles avec des mots tel que 'mais', 'donc', 'or', 'par contre', 'en revanche', 'en effet', 'cependant', 'toutefois', 'par ailleurs', 'par contre', 'par contre, 'enfin'"},
            {"role": "user",
            "content": "Voici la liste des actualités à synthétiser : " + text},
        ],
        max_tokens=200,
        temperature=0.9,
    )

    try:
        return response.choices[0].message.content
    except:
        print('Une erreur est survenue avec la commande /actu')
        print(response)
        return 'Une erreur est survenue'

def handleJson(prompt) -> str:
    # check if prompt is an url
    if not prompt.startswith('http'):
        return 'L\'url n\'est pas valide'
    
    # get html from url
    response = requests.get(prompt).text
    soup = BeautifulSoup(response, "html.parser")
    html_text = soup.body

    # remove script, style, head, header, footer, iframe, canvas, noscript, form
    for tag in html_text(["script", "style", "head", "header", "footer", "iframe", "canvas", "noscript", "form"]):
        tag.decompose()

    html_text = html_text.text.replace("\n", " ").replace("\t", " ")
    html_text = html_text[:5000]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
            "content": "Tu es un expert web et json, tu dois trouver dans le html les artciles ou données et me les rendre sous format json impérativement"},
            {"role": "user",
            "content": html_text},
        ],
        temperature=0.2,
    )

    try:
        return response.choices[0].message.content
    except:
        print('Une erreur est survenue avec la commande /json')
        print(response)
        return 'Une erreur est survenue'