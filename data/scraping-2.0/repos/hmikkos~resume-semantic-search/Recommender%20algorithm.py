# imports
import pandas as pd
import pickle
import os
import openai

from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)
OPEN_API_KEY = ""
openai.api_key = OPEN_API_KEY

# constants
EMBEDDING_MODEL = "text-embedding-ada-002"

#LOAD DATA
#creates a pandas dataframe df containing all the texts contained in the text files in the folder CV_parsed


import os
import pandas as pd
import json
import re

def create_dataframe(folder):
    files = os.listdir(folder)
    data = []
    index = []
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(folder, file), 'r') as f:
                text = f.read()
                json_text = json.loads(text)
                
                # remove unwanted fields
                json_text['info_basique'].pop('email', None)
                json_text['info_basique'].pop('linkedin_url', None)
                json_text['info_basique'].pop('annee_de_diplomation', None)
                for exp in json_text['experience_professionnelle']:
                    exp.pop('durée', None)

                # convert the dictionary back into a JSON string
                json_text = json.dumps(json_text)
                data.append(json_text)


                # extract index from filename
                match = re.search(r'\d+', file)
                if match:
                    index.append(int(match.group()))

    df = pd.DataFrame(data, columns=['description'], index=index)
    return df

df = create_dataframe('CV_parsed')


#remove the emails, names and linkedin links from the text

# establish a cache of embeddings to avoid recomputing
# cache is a dict of tuples (text, model) -> embedding, saved as a pickle file

# set path to embedding cache
embedding_cache_path = "data/recommendations_embeddings_cache.pkl"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file) 

# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


#name the index column : index
df.index.name = 'title'


def print_recommendations_from_strings(
    strings: list[str],
    index_of_source_string: int,
    k_nearest_neighbors: int = 1,
    model=EMBEDDING_MODEL,
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all strings
    embeddings = [embedding_from_string(string, model=model) for string in strings]
    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # print out source string
    query_string = strings[index_of_source_string]
    print(f"Source string: {query_string}")
    # print out its k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_string == strings[i]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # print out the similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i][0:50]}
        Distance: {distances[i]:0.3f}"""
        )

    return indices_of_nearest_neighbors


index_source = 0
source_string = "Les principales responsabilités du Responsable Communication Externe et Digitale se résument comme suit : Définition de la politique de communication externe et digitale Groupe : Participer à l'élaboration de la politique Communication Externe Groupe (communication digitale, relations publiques, relation presse et médias sociaux, évènementiel, sponsoring …) Contribuer activement à l'élaboration de la stratégie de communication externe et digitale Assurer la traduction de la stratégie de communication externe et digitale en plan d'action annuel. Assurer la planification et la budgétisation du plan d'action annuel. Déploiement de la stratégie de communication externe et digitale Groupe : Piloter la réalisation des campagnes de communication. Superviser la conception et la production de supports de communication destinés au public externe (brochures, kits, affiches. etc.) Veiller au développement d’outils de communication innovants sur de nouveaux canaux de communication (réseaux sociaux, mobile, site web…etc) Superviser la coordination et les négociations avec les prestataires de service ou agences de communication ainsi que les médias. Animation de la communication externe et digitale : Rédaction d’articles, communiqués de presse, dossiers de presse à transmettre aux journalistes. Animer la revue de presse et le press-book de la CDG. Participer à la conception, superviser la réalisation et le webmastering du site web. Assurer l'organisation, l'animation et la gestion des évènements (conférence de presse, forums, …etc). Réaliser et gérer les outils de communication (plaquette institutionnelle, site web, newsletter…) de la CDG. Animer les échanges avec les followers/ abonnés sur les réseaux sociaux de la CDG. Fidéliser la communauté d’internautes propre à la CDG au niveau des réseaux sociaux. Assurer la gestion des sponsorings et des partenariats de la CDG. Gérer la communication financière de la CDG (Publications, CP, …). Veille et amélioration continue : Assurer une veille de son secteur d’activité sur l'ensemble des canaux de communication. Analyser les résultats des différentes campagnes de communication. Réaliser un benchmark des stratégies concurrentes, ce qui permettra d’évaluer son positionnement sur les réseaux et de se démarquer. Assurer la veille sur sa marque, la définition de stratégies et la formalisation de procédures. Management d'équipe : Définir le plan d’actions annuel de la structure gérée et suivre les indicateurs périodiques de sa réalisation. Valider l’organisation des structures gérées dans le respect de la politique et des procédures des ressources humaines de l’entreprise.Mettre en place les plans de développement des compétences de ses collaborateurs et préserver un bon niveau de motivation de ses équipes. Garantir l’amélioration continue de la qualité des prestations des structures gérées. Profil recherché : De formation supérieure de niveau Bac+5 d’une grande école de commerce ou université. Expérience professionnelle de 8 ans minimum dans le domaine de la communication institutionnelle dans un établissement financier, un grand groupe national ou international, dont 3 ans minimum dans un poste de management, Compétences requises pour le poste : Maîtrise des techniques de veille et des nouvelles tendances. Connaissances approfondies de l’univers média et hors média. Maîtrise des logiciels de bureautique et de publication. Excellentes capacités rédactionnelles. Parfaite maîtrise de la langue arabe, française et anglaise. Sens de la rigueur et de la responsabilité. Autonomie et bon sens de l’organisation. Prestance et aisance relationnelle. "

#add the source string to the dataframe with the index_source, and put it in the beginning of the dataframe
df.loc[index_source] = source_string
df = df.sort_index()




candidates_descriptions = df["description"].tolist()

ACE_candidates = print_recommendations_from_strings(
    strings=candidates_descriptions,  # let's base similarity off of the candidate description
    index_of_source_string=0,  # let's look at candidates similar to the first one about Tony Blair
    k_nearest_neighbors=50,  # let's look at the 30 most similar candidates
)


# create labels for the recommended articles
def nearest_neighbor_labels(
    list_of_indices: list[int],
    k_nearest_neighbors: int = 5
) -> list[str]:
    """Return a list of labels to color the k nearest neighbors."""
    labels = ["Other" for _ in list_of_indices]
    source_index = list_of_indices[0]
    labels[source_index] = "Source"
    for i in range(k_nearest_neighbors):
        nearest_neighbor_index = list_of_indices[i + 1]
        labels[nearest_neighbor_index] = f"Nearest neighbor (top {k_nearest_neighbors})"
    return labels

"""
ACE_labels = nearest_neighbor_labels(ACE_candidates, k_nearest_neighbors=5)


# get embeddings for all article descriptions
embeddings = [embedding_from_string(string) for string in candidates_descriptions]
# compress the 2048-dimensional embeddings into 2 dimensions using t-SNE
tsne_components = tsne_components_from_embeddings(embeddings)




# a 2D chart of nearest neighbors of the Tony Blair article
chart = chart_from_components(
    components=tsne_components,
    labels=ACE_labels,
    strings=candidates_descriptions,
    width=600,
    height=500,
    title="Nearest neighbors of the ACE",
    category_orders={"label": ["Other", "Nearest neighbor (top 5)", "Source"]},
)

import matplotlib.pyplot as plt
import plotly.express as px
chart.show()"""