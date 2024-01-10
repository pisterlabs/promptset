# %%
# CSV Files downloaded from https://www.data.gouv.fr/fr/datasets/repertoire-national-des-associations/  Fichier RNA Waldec du 01 Mars 2022
import datetime as dt
import glob
import os
import time

import boto3
import numpy as np
import openai
import pandas as pd
import requests_cache
from diskcache import Cache
from geopy.geocoders import Nominatim
from lambdaprompt import GPT3Prompt
from pandarallel import pandarallel
from rich.console import Console

# %%
start = time.time()
file_location = os.getcwd() + "/rna_waldec_20220301/"
all_files = glob.glob(os.path.join(file_location, "*.csv"))

columns = [
    "id",
    "titre",
    "objet",
    "objet_social1",
    "objet_social2",
    "adrs_numvoie",
    "position",
    "adrs_typevoie",
    "adrs_libvoie",
    "adrs_codepostal",
    "adrs_libcommune",
    "siteweb",
]
columns = [
    "id",
    "titre",
    "objet",
    "objet_social1",
    "objet_social2",
    "adrs_numvoie",
    "position",
    "adrs_typevoie",
    "adrs_libvoie",
    "adrs_codepostal",
    "adrs_libcommune",
    "siteweb",
]

df = pd.concat(
    (
        pd.read_csv(
            f,
            delimiter=";",
            header=0,
            encoding="ISO-8859-1",
            usecols=columns,
            engine="c",
        )
        for f in all_files
    ),
    ignore_index=True,
)
df_associations = pd.concat(
    [
        pd.read_csv(
            f,
            delimiter=";",
            header=0,
            encoding="ISO-8859-1",
            usecols=columns,
            engine="c",
        )
        for f in all_files
    ],
    ignore_index=True,
)

end = time.time()
print(f"Time to read all CSV : {dt.timedelta(seconds=end - start)}")

# %%
ssm = boto3.client("ssm", region_name="eu-central-1")
ssm = boto3.client("ssm", region_name="eu-central-1")

openai.api_key = ssm.get_parameter(
    Name="/tchoung-te/openai_api_key", WithDecryption=False
)["Parameter"]["Value"]
    Name="/tchoung-te/openai_api_key", WithDecryption=False
)["Parameter"]["Value"]


# setter la variable d'environnement
os.environ["OPENAI_API_KEY"] = openai.api_key


# %%
start = time.time()


def filter_cameroon(df):
    return df[
        df["titre"].str.contains("CAMEROUN", case=False, na=False)
        | df["objet"].str.contains("CAMEROUN", case=False, na=False)
        | df["titre"].str.contains("KMER", case=False, na=False)
        | df["objet"].str.contains("KMER", case=False, na=False)
    ]
    """
    Filter associations with "Cameroun" in the title or the object
    """
    return df[
        df["titre"].str.contains("CAMEROUN", case=False, na=False)
        | df["objet"].str.contains("CAMEROUN", case=False, na=False)
    ]


def remove_closed(df):
    """
    Remove closed associations
    """
    return df[df["position"].str.contains("D|S") == False]


def normalize(df):
    df["titre"] = df["titre"].str.upper()
    df["objet"] = df["objet"].str.lower()
    df["adrs_codepostal"] = df["adrs_codepostal"].astype(int)
    df["objet_social1"] = df["objet_social1"].astype(int)
    df["objet_social2"] = df["objet_social2"].astype(int)
    """
    Normalize strings in the associations infos
    """
    df["titre"].str.upper()
    df["objet"].str.lower()
    df["adrs_codepostal"] = df["adrs_codepostal"].astype(int)
    df["objet_social1"] = df["objet_social1"].astype(int)
    df["objet_social2"] = df["objet_social2"].astype(int)
    # this will avoid nan in adrs which concatenate multiple values
    df = df.fillna("")

    return df


def select_relevant_columns(df):
    return df[
        [
            "id",
            "titre",
            "objet",
            "objet_social1",
            "objet_social2",
            "adrs_numvoie",
            "adrs_typevoie",
            "adrs_libvoie",
            "adrs_codepostal",
            "adrs_libcommune",
            "siteweb",
        ]
    ]


df2 = df.pipe(filter_cameroon).pipe(remove_closed).pipe(normalize)
df_cameroon_associations = (
    df_associations.pipe(filter_cameroon).pipe(remove_closed).pipe(normalize)
)

end = time.time()
print(f"Time to Filter Rows : {dt.timedelta(seconds=end - start)}")

# %%
text_prompt = """
Normalize the addresses in french.
Don't ignore any lines and treat each address separetely and go step by step
For the result, follow the same order I give you. Be sure that the number of lines you generate equals to the ones I pass you

Adresses:
62 RUE de la Ramassière 1600 Reyrieux
20 bis RUE de Lyon 1800 Meximieux
   1340 Marsonnas
2 RUE Lalande 1000 Bourg-en-Bresse
2 BD Irène Joliot Curie 1000 Bourg-en-Bresse
  2, rue de la République 1000 Bourg-en-Bresse
27 RUE Lulli 1000 Saint-Denis-lès-Bourg
331 CHEM1 de Tir-Mir 1220 Divonne-les-Bains
7 CHEM1 du levant 1220 Divonne-les-Bains
   1210 Ferney-Voltaire
429 RUE de l'Europe 1630 Saint-Genis-Pouilly
177 RUE du Commerce 1170 Gex
232 RUE du Muret 1200 Châtillon-en-Michaille
11 PL Aristide Briand 2130 Fère-en-Tardenois
5 RUE du Garet 69001 Lyon
23 BD Gambetta 2700 Tergnier
1 RUE de l'Eglise 2220 Cuiry-Housse
   3350 Le    Brethon
5/7 IMP Dieudonné Coste 3000 Moulins
   4000 Digne-les-Bains
1 PL des Félibres 4130 Volx
775 CHEM1 du mas de Bos à la Font de Rey 30300 Beaucaire
17 VILL1 Curial 75019 Paris
 CHEM1 de Pré Lacour 5140 Aspremont
50 AV du Commandant Bret 6400 Cannes

Corrections:

62 Rue de la Ramassière, 1600 Reyrieux
20 bis Rue de Lyon, 1800 Meximieux
1340 Marsonnas
2 Rue Lalande, 1000 Bourg-en-Bresse
2 Boulevard Irène Joliot Curie, 1000 Bourg-en-Bresse
2 Rue de la République, 1000 Bourg-en-Bresse
27 Rue Lulli, 1000 Saint-Denis-lès-Bourg
331 Chemin de Tir-Mir, 1220 Divonne-les-Bains
7 Chemin du Levant, 1220 Divonne-les-Bains
1210 Ferney-Voltaire
429 Rue de l'Europe, 1630 Saint-Genis-Pouilly
177 Rue du Commerce, 1170 Gex
232 Rue du Muret, 1200 Châtillon-en-Michaille
11 Place Aristide Briand, 2130 Fère-en-Tardenois
5 Rue du Garet, 69001 Lyon
23 Boulevard Gambetta, 2700 Tergnier
1 Rue de l'Eglise, 2220 Cuiry-Housse
3350 Le Brethon
5/7 Impasse Dieudonné Coste, 3000 Moulins
4000 Digne-les-Bains
1 Place des Félibres, 4130 Volx
775 Chemin du Mas de Bos à la Font de Rey, 30300 Beaucaire
17 Villa Curial, 75019 Paris
Chemin de Pré Lacour, 5140 Aspremont
50 Avenue du Commandant Bret, 6400 Cannes

###
Adresses:
{{ prmt }}

Corrections:
"""
prompt_function_batch = GPT3Prompt(text_prompt, max_tokens=2000)

console = Console()
cache = Cache("prompt_address_cache")

"""
OpenAI has a limit of 20 requests per minute and 150 000 TPM and lambdaprompt is hardcoded to 500 tokens max
So we need to batch request by 25 max 25*18[mean token adrs size] = 450 with a proper delay
"""

# Build a list of all adresses in the cache & remove useless spaces
all_adresses = "\n".join(list(cache))
all_adresses = all_adresses.split("\n")
all_adresses = "\n".join(list(cache))
all_adresses = all_adresses.split("\n")
all_adresses = [x.strip() for x in all_adresses]

# Build adresse by concatenation
df2["adrs"] = (
    df2["adrs_numvoie"].map(str)
    + " "
    + df2["adrs_typevoie"].map(str)
    + " "
    + df2["adrs_libvoie"].map(str)
    + " "
    + df2["adrs_codepostal"].map(str)
    + " "
    + df2["adrs_libcommune"].map(str)
)
df_cameroon_associations["adrs"] = (
    df_cameroon_associations["adrs_numvoie"].map(str)
    + " "
    + df_cameroon_associations["adrs_typevoie"].map(str)
    + " "
    + df_cameroon_associations["adrs_libvoie"].map(str)
    + " "
    + df_cameroon_associations["adrs_codepostal"].map(str)
    + " "
    + df_cameroon_associations["adrs_libcommune"].map(str)
)


df_cameroon_associations["adrs"] = df_cameroon_associations.adrs.apply(
    lambda x: x.strip()
)


# Filter only adresses not present in the cache yet
df_not_in_cache = df_cameroon_associations[
    ~df_cameroon_associations.adrs.isin(all_adresses)
]

print(f"{len(df_not_in_cache)} adresses not present in cache...")
# %%
if len(df_not_in_cache) > 0:
    num_batches = int(np.ceil(len(df_not_in_cache) / 25))
    batches = np.array_split(df_not_in_cache, num_batches)

    for id_batch, batch in enumerate(batches):
        print(f"ID Batch : {id_batch}")
        list_adresses = " "
        for address in batch["adrs"]:
            list_adresses += f"{address}\n"

        if list_adresses not in list(cache):
            clean_adresses = prompt_function_batch(prmt=list_adresses)
            clean_adresses = clean_adresses.split("\n")
            cache[list_adresses] = (
                clean_adresses[1:]
                if len(clean_adresses) != len(batch["adrs"])
                else clean_adresses
            )
            time.sleep(120)
        batch["adrs"] = cache[list_adresses]

# %%
# Downloaded from https://download.geonames.org/export/zip/
region_by_postal_codes = pd.read_csv(
    "code-postal-geonames.tsv", delimiter="\t", index_col=1
).to_dict()["REGION"]

dept_by_postal_codes = pd.read_csv(
    "code-postal-geonames.tsv", delimiter="\t", index_col=1
).to_dict()["DEPT"]

region_by_postal_codes["97300"] = "Guyane"
dept_by_postal_codes["97300"] = "Guyane"

region_by_postal_codes["97419"] = "Réunion"
dept_by_postal_codes["97419"] = "Réunion"

region_by_postal_codes["97438"] = "Réunion"
dept_by_postal_codes["97438"] = "Réunion"

region_by_postal_codes["97600"] = "Mayotte"
dept_by_postal_codes["97600"] = "Mayotte"


waldec_csv = pd.read_csv(
    "rna-nomenclature-waldec.csv", delimiter=";", index_col=2
).to_dict()["Libellé objet social parent"]
# pprint(waldec_csv)
waldec_csv[00000] = "AUTRES"
waldec_csv[19060] = "INTERVENTIONS SOCIALES"
waldec_csv[22010] = "REPRÉSENTATION, PROMOTION ET DÉFENSE D'INTÉRÊTS ÉCONOMIQUES"
waldec_csv[6035] = "CULTURE, PRATIQUES D'ACTIVITÉS ARTISTIQUES, PRATIQUES CULTURELLES"
waldec_csv[40510] = "ACTIVITÉS RELIGIEUSES, SPIRITUELLES OU PHILOSOPHIQUES"
waldec_csv[17035] = "SANTÉ"
waldec_csv[40580] = "ACTIVTÉS RELIGIEUSES, SPIRITUELLES OU PHILOSOPHIQUES"


# %%

def get_dept_region(code_postal):
    try:
        dept = dept_by_postal_codes[str(code_postal)]
    except KeyError:
        dept = dept_by_postal_codes[
            [x for x in dept_by_postal_codes.keys() if str(code_postal) in x][0]
        ]

    try:
        region = region_by_postal_codes[str(code_postal)]
    except KeyError:
        region = region_by_postal_codes[
            [x for x in region_by_postal_codes.keys() if str(code_postal) in x][0]
        ]

    return {"dept": dept, "region": region}


def add_dept_and_region(df):
    """
    Add department and region to list of associations
    """

    df["dept"] = df.apply(
        lambda row: get_dept_region(row["adrs_codepostal"])["dept"], axis=1
    )
    df["region"] = df.apply(
        lambda row: get_dept_region(row["adrs_codepostal"])["region"], axis=1
    )

    return df


def add_social_object_libelle(df):
    df["social_object1_libelle"] = df.apply(
        lambda row: waldec_csv[int(row["objet_social1"] or 0)], axis=1
    )
    df["social_object2_libelle"] = df.apply(
        lambda row: waldec_csv[int(row["objet_social2"] or 0)], axis=1
    )

    return df


df_cameroon_associations = df_cameroon_associations.pipe(add_dept_and_region).pipe(
    add_social_object_libelle
)

# df2["dept"] = df2.apply(lambda row: get_dept_region(row["adrs_codepostal"])["dept"],axis=1)
# df2["region"] = df2.apply(lambda row: get_dept_region(row["adrs_codepostal"])["region"],axis=1)

# get_info("W212001727")
# get_dept_region(30913)

# %%
pandarallel.initialize(progress_bar=True)
requests_cache.install_cache("geocode_cache")

geolocator = Nominatim(user_agent="tchoung-te.mongulu.cm")


def add_lat_lon(df):
    def geocode(adrs):
        return geolocator.geocode(adrs, country_codes="fr", timeout=None)

    df["geocode"] = df.parallel_apply(lambda row: geocode(row["adrs"]), axis=1)

    # df2[df2.geocode.isnull()] 417 rows n'ont pas de géocodage
    # df2[df2.adrs_libcommune.isnull()] # mais aucune n'a le nom de la commune vide donc ce sera ça qui sera utilisé
    df.loc[df.geocode.isnull(), "geocode"] = df[df.geocode.isnull()].parallel_apply(
        lambda row: geocode(row["adrs_libcommune"]), axis=1
    )
    # 4 associations restent sans geocode, on les force
    df.loc[df.id == "W251002897", "geocode"] = df[df.geocode.isnull()].apply(
        lambda row: geocode("Besançon"), axis=1
    )
    df.loc[df.id == "W302012059", "geocode"] = df[df.geocode.isnull()].apply(
        lambda row: geocode("Nimes"), axis=1
    )
    df.loc[df.id == "W313015063", "geocode"] = df[df.geocode.isnull()].apply(
        lambda row: geocode("Toulouse"), axis=1
    )
    df.loc[df.id == "W382000558", "geocode"] = df[df.geocode.isnull()].apply(
        lambda row: geocode("Satolas-et-Bonce"), axis=1
    )

    df["longitude"] = df["geocode"].apply(lambda x: x.longitude)
    df["latitude"] = df["geocode"].apply(lambda x: x.latitude)
    df = df.drop(columns=["objet_social1", "objet_social2", "geocode"], axis=1)

    return df


def format_libelle_for_gogocarto(df):
    # Gogocarto lit une liste de catégories sur un champ défini et considère la virgule comme le caractère de séparation
    # On a donc opté pour remplacer la virgule(",") par le slash("/")
    df["social_object1_libelle"] = df["social_object1_libelle"].apply(
        lambda x: x.replace(",", "/")
    )
    df["social_object2_libelle"] = df["social_object2_libelle"].apply(
        lambda x: x.replace(",", "/")
    )

    return df


df_cameroon_associations = df_cameroon_associations.pipe(add_lat_lon).pipe(
    format_libelle_for_gogocarto
)

# %%


def remove_space_at_the_end(x: str):
    if x is not None:
        return x.strip()


def replace_double_quote(x: str):
    if x is not None:
        return x.replace('""', "'")


def normalize_final(data: pd.DataFrame):
    text_columns = [
        "titre",
        "objet",
        "social_object1_libelle",
        "social_object2_libelle",
    ]
    data[text_columns] = data[text_columns].apply(
        lambda x: x.apply(remove_space_at_the_end)
    )
    data[text_columns] = data[text_columns].apply(
        lambda x: x.apply(replace_double_quote)
    )
    data["titre"] = data["titre"].apply(lambda x: x.upper())
    data["objet"] = data["objet"].apply(lambda x: x.lower())

    return data


df_cameroon_associations = df_cameroon_associations.pipe(normalize_final)


# %%
df_cameroon_associations.to_csv("rna-real-mars-2022-new.csv")


print(f"{len(df_cameroon_associations)} Cameroon Associations ready!!")
