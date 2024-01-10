import os
import re
import openai

from mediaLab.ImageSuggester import generate_img
from mediaLab.main import download_video

openai.api_key = os.environ["OPENAI_API_KEY"]

def find_video_path(keywords, folder, id_l):
    path = download_video(keywords, folder, id_l)
    # Ici, vous pouvez insérer votre logique pour trouver le chemin d'une vidéo en fonction des mots clés
    return path

def clean_video_path(video_path):
    # Utilise une expression régulière pour trouver tout ce qui se termine par .mp4
    match = re.search(r'(.*\.mp4)', video_path)
    if match:
        return match.group(1)
    return video_path

def replace_keywords_with_video_path(match, folder, id_l):
    elements_str = match.group(1)
    elements = elements_str.split(',')
    parsed_elements = []
    for element in elements:
        element = element.strip()
        print("on va généééééérerrrrr sur DALEEEEEE")
        video_path = generate_img(element)
    #     try:
    #         parsed_elements.append(int(element))
    #     except ValueError:
    #         parsed_elements.append(element)
    # video_path, id_l = find_video_path(parsed_elements, folder, id_l)
    # if not video_path:
    #     print("on va généééééérerrrrr sur DALEEEEEE")
    #     video_path = generate_img(parsed_elements[0])
    cleaned_video_path = clean_video_path(video_path)
    return cleaned_video_path, id_l

def parse_elements(elements_str):
    elements = elements_str.split(',')
    parsed_elements = []
    for element in elements:
        element = element.strip()
        try:
            parsed_elements.append(int(element))
        except ValueError:
            parsed_elements.append(element)
    return parsed_elements


def find_words_and_list_in_string(s):
    pattern = r'\[(.*?)\]'
    matches = [(m.end(), m.group(1)) for m in re.finditer(pattern, s)]
    results = []

    end_idx = 0
    for match_end, elements in matches:
        # Extract up to 3 words after the match
        post_match_str = s[match_end:].strip().split()[:3]

        # Save the list and the words into a single list and append to results
        parsed_elements = parse_elements(elements)
        combined_list = [parsed_elements] + post_match_str
        results.append(combined_list)

    return results

# print(find_words_and_list_in_string(text)[0][0])
def mediaSuggester(script_video):

    prompt = f"""
    Je souhaite automatiser la création de vidéos en utilisant le script pré-écrit suivant {script_video}.
     Pour chaque phrase du script, je souhaite que vous génériez une liste Python de courtes descriptions en qui
      correspondent à des situations générales de la vie qui illustre la phrase qui pourraient être trouvées dans 
      une base de données vidéo. Ces mots-clés serviront à rechercher des vidéos pertinentes pour chaque segment
       du script. Il est impératif que la réponse soit formatée de manière très spécifique. Chaque liste de mots-clés 
       doit précéder immédiatement la phrase à laquelle elle se rapporte, séparée par un espace. Le format de la réponse 
       doit être du texte brut (plain text) et non du code. Par exemple : ["man plays with a dog", "happy dog rolls in
        the mud", "happy dog"] Voici la première phrase du script. ["mot-clé4", "mot-clé5", "mot-clé6"] Voici la
         deuxième phrase du script.
    """
    print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tu es un programme informatique"},
            {"role": "user", "content": prompt}
        ]
    )

    script_text = response['choices'][0]['message']['content']
    print(script_text)


def script_stroke(folder, downloaded_ids):
    chemin_script = folder + "script.txt"
    chemin_save = folder + "save.txt"

    with open(chemin_save, 'r', encoding='utf-8') as fichier:
        text = fichier.read()

    updated_downloaded_ids = downloaded_ids.copy()  # Créer une copie pour éviter de modifier la liste originale

    def replace_function(match):
        nonlocal updated_downloaded_ids
        cleaned_video_path, updated_downloaded_ids = replace_keywords_with_video_path(match, folder, updated_downloaded_ids)
        return cleaned_video_path

    script_text = re.sub(r'\[(.*?)\]', replace_function, text)
    with open(chemin_script, 'w', encoding='utf-8') as fichier:
        fichier.write(script_text)
    # with open(chemin_save, 'w', encoding='utf-8') as fichier:
    #     fichier.write(text)

    return updated_downloaded_ids



# mediaSuggester()