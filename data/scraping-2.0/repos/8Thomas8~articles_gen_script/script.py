import os
import openai
import json
from tqdm import tqdm
from slugify import slugify
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

# Listes des catégories
categories = json.loads(os.getenv("ARRAY_CATEGORIES"))

# Nombre d'articles par catégorie
articleNumberPerCategory = int(os.getenv("NUMBER_ARTICLES_PER_CATEGORY"))

# Date des articles
date = os.getenv("ARTICLES_DATE")

# Initialiser une session OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_content(prompt):
    completions = openai.Completion.create(
        engine=os.getenv("MODEL_ENGINE"),
        prompt=prompt,
        max_tokens=int(os.getenv("MAX_TOKENS")),
        n=1,
        stop=None,
        temperature=float(os.getenv("TEMPERATURE")),
    )

    message = completions.choices[0].text
    return message

# Boucles sur les catégories
for category in tqdm(categories, desc="Catégories"):

    # Créé un dossier avec le nom de la catégorie
    directory = "articles/" + category.replace(" ", "_")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Générer les articles dans le dossier
    for i in tqdm(range(articleNumberPerCategory), desc="Articles" + category ):
        try:
            title = generate_content(os.getenv("PROMPT_TITLE") + category + '.')
        except openai.error.InvalidRequestError as e:
            print("Une erreur s'est produite :", e)
        try:
            content = generate_content(os.getenv("PROMPT_CONTENT") + title + ".")
        except openai.error.InvalidRequestError as e:
            print("Une erreur s'est produite :", e)
        try:
            content = generate_content(os.getenv("PROMPT_CONTENT_BIS") + content)
        except openai.error.InvalidRequestError as e:
            print("Une erreur s'est produite :", e)
        try:
            excerpt = generate_content(os.getenv("MODEL_ENGIPROMPT_EXCERPT_1NE") + content + os.getenv("PROMPT_EXCERPT_2") + title + '.')
        except openai.error.InvalidRequestError as e:
            print("Une erreur s'est produite :", e)
        try:  
            meta_description = generate_content(os.getenv("PROMPT_META_DESC_1") + content + os.getenv("PROMPT_META_DESC_2") + title +'.')
        except openai.error.InvalidRequestError as e:
            print("Une erreur s'est produite :", e)
        
        # Ecriture du fichier
        filename = slugify(title) + ".md"

        with open(directory + "/" + filename, "w", encoding="utf-8") as f:
            f.write("---"+ "\n")
            f.write("layout: 'blog-article'"+ "\n")
            f.write('title: "' + title.replace("\n", "").replace("\"", "") + '"' + "\n")
            f.write('chapeau: "' + excerpt.replace("\n", "").replace("\"", "") + '"' + "\n")
            f.write('description: "' + meta_description.replace("\n", "").replace("\"", "") + '"' + "\n")
            f.write('image:'+ "\n")
            f.write('   src: /images/blog/' + slugify(category) + '/' + slugify(title) + '.png'+ "\n")
            f.write('   alt: "' + title.replace("\n", "").replace("\"", "") + '"' + "\n")
            f.write('createdAt: ' + date + "\n")
            f.write('updatedAt: ' + date + "\n")
            f.write("isArticle: 'true'" + "\n")
            f.write("---" + "\n\n")
            f.write('<div class="mt-4 rounded-md bg-gray-100 p-4">' + "\n")
            f.write("Sommaire :" + "\n\n")
            f.write('<ol class="flex flex-col">' + "\n")
            f.write('<li><a href="" title=""></a></li>' + "\n")
            f.write("</ol>" + "\n")
            f.write("</div>" + "\n\n")
            f.write(content + "\n")
