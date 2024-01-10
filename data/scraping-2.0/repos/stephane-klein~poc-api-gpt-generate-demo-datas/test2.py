import os
import json
import pprint
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

output = []

for d in range(1, 8):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Génère un billet de blog d'environ 40 mots, coupé entre 2 et 3 paragraphes (entourés de balises <p>), qui parle d'un randonneur nommé John Doe qui fait le Chemin de Chemin de Stevenson accompagné par un yak, à partir du 12 juin 2023. Le billet parle de la journée d'étape %s.\nLe résultat doit être au format json, le titre dans le champ 'title', le corps dans le champ 'body'. Il doit aussi contenir entre 2 et 6 tags, dans le champ 'tags'. La date du jour de randonné dans le champ 'date', au format YYYY-MM-DD, en incrémentant le jour avec le numéro d'étape." % d,
        temperature=1,
        max_tokens=3000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    try:
        output.append(json.loads(response['choices'][0]['text'], strict=False))
    except:
        print(response['choices'][0]['text'])

print(json.dumps(output, indent=4, ensure_ascii=False))
