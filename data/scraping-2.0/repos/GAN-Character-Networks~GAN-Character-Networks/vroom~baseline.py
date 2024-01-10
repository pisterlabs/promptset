r"""  Pipeline for cooccurences extraction with baseline methods

Authors
--------
 * Nicolas Bataille 2023
 * Adel Moumen 2023, 2024
 * Gabriel Desbouis 2023
"""
import json

from openai import OpenAI

from vroom.alias import get_aliases_fuzzy_partial_token
from vroom.cooccurences import get_cooccurences
from vroom.loggers import JSONLogger
from vroom.NER import (
    chunk_text_by_sentence,
    get_entities_from_file,
    get_positions_of_entities,
    tag_text_with_entities,
)


def get_cooccurences_from_text(path: str):
    """
    Get the coocurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """

    entities, chunks = get_entities_from_file(path)
    return get_cooccurences(chunks, entities)


def find_cooccurences_aliases(cooccurences, aliases):
    no_alias_1_list = []
    no_alias_2_list = []
    cooccurrences_aliases = []

    for cooc in cooccurences:
        no_alias_1 = True
        no_alias_2 = True

        for alias in aliases:
            if cooc[0] in alias:
                no_alias_1 = False
            if cooc[1] in alias:
                no_alias_2 = False

        if no_alias_1:
            no_alias_1_list.append(cooc[0])
        if no_alias_2:
            no_alias_2_list.append(cooc[1])

    print("no alias 1 list: ", no_alias_1_list)
    print("no alias 2 list: ", no_alias_2_list)

    for cooccurence in cooccurences:
        cooc_1_aliases = [
            alias
            for alias in aliases
            if cooccurence[0].lower() in [a.lower() for a in alias]
        ]
        cooc_2_aliases = [
            alias
            for alias in aliases
            if cooccurence[1].lower() in [a.lower() for a in alias]
        ]

        if (
            cooc_1_aliases
            and cooc_2_aliases
            and cooc_1_aliases[0] != cooc_2_aliases[0]
        ):
            cooccurrences_aliases.append((cooc_1_aliases[0], cooc_2_aliases[0]))

    print("aliases: ", aliases)
    return cooccurrences_aliases


def get_cooccurences_with_aliases(path: str, logger: JSONLogger = None):
    """
    Get the aliases of the cooccurences of characters from the given text.

    Args:
        path (str): The path of the text file.
        logger (JSONLogger, optional): The logger to save the aliases. Defaults to None.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """
    entities, chunks = get_entities_from_file(path)
    cooccurences = get_cooccurences(chunks, entities)
    entities_unfold = [entity for sublist in entities for entity in sublist]
    aliases = get_aliases_fuzzy_partial_token(entities_unfold, 99)

    if logger is not None:
        saves = {}
        for i, (chunk, entity) in enumerate(zip(chunks, entities)):
            saves[f"chunk_{i}"] = {
                "chunk": chunk,
                "entities": entity,
            }
        saves["entities"] = list(
            set([entity["word"] for entity in entities_unfold])
        )
        saves["aliases"] = aliases
        saves["cooccurences"] = cooccurences
        logger(saves)

    cooccurences_aliases = find_cooccurences_aliases(cooccurences, aliases)
    return cooccurences_aliases


def get_cooccurences_with_aliases_and_gpt(path: str, logger: JSONLogger = None):
    """
    Get the aliases of the cooccurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """
    system_prompt = r"""
Tu es un expert dans les livres "La Fondation" de Isaac Asimov.
Ton but est d'à partir d'une liste de personnes de déterminer qui est qui en faisant un regroupement. Chaque regroupement représente une personne avec toutes ses références.

Pour réaliser ce regroupement, base toi sur la sémantique des mots, notamment le genre, les titres, les ressemblances typographiques, etc.

Donne ta réponse sous le format JSON suivant, et ne dévie pas de cette tâche :

{
   '0' : [reference_1, …],
   '1' : [reference_1, …],
}

Chaque entrée du JSON correspond à un personnage et à l'ensemble de ses références. La clef est un chiffre qui donne la position dans le JSON. La position n’a pas d’importance.

Voici des exemples pour t'aider :

Example 1 :

Input:
CLÉON Ier
Empereur
Cléon
Sire
l'Empereur
l'empereur Cléon
Hari Seldon
Seldon
Eto Demerzel
Demerzel
Lieutenant Alban Wellis
Wellis
Hummin
Trantor

Output:
{
    '0' : ['CLÉON Ier', 'Empereur', 'Cléon', 'Sire', 'l'Empereur', 'l'empereur Cléon'],
    '1' : ['Hari Seldon', 'Seldon', 'Eto Demerzel', 'Demerzel'],
    '2' : ['Lieutenant Alban Wellis', 'Wellis'],
    '3' : ['Hummin'],
}

Tu vas desormais recevoir une liste de personnes en input, et tu dois les regrouper s'ils se réfèrent à la même personne. Tu n'as pas le droit d'inventer de nouveaux personnages et tu dois uniquement utiliser cette liste. Les noms ne doivent pas être modifiés, y compris les espaces, apostrophes, etc. Si tu ne respectes pas ces règles, tu seras fortement pénalisé.
Tu as le droit de supprimer des personnes si tu penses qu'ils ne sont pas des personnages du livre. En effet, le système qui prédit un personnage peut se tromper, et tu dois le corriger.

Fais-le pour les personnes suivantes et essaie de trouver les meilleurs regroupements possibles. Je compte sur toi, merci !
    """

    entities, chunks = get_entities_from_file(path, device="cuda")
    cooccurences = get_cooccurences(chunks, entities)
    entities = [entity for sublist in entities for entity in sublist]
    word_entities = [entity["word"] for entity in entities]
    print("entities = ", set(word_entities))
    user_prompt = """

    Input :
    """
    user_prompt += "\n".join(set(word_entities)) + "\nOutput : "
    experiment_details = """
    gpt-4-1106-preview avec 1 exemples de donné en prompt.
    """  # GPT-3.5-turbo-1106

    client = OpenAI()
    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-4-1106-preview",  # gpt-4-1106-preview
    }
    response = client.chat.completions.create(
        **params,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    generated_content = response.choices[0].message.content
    generated_content = json.loads(generated_content)

    aliases = []
    for key in generated_content:
        aliases.append(generated_content[key])

    if logger is not None:
        saves = {}
        saves["experiment_details"] = experiment_details
        for i, (chunk, entity) in enumerate(zip(chunks, entities)):
            saves[f"chunk_{i}"] = {
                "chunk": chunk,
                "entities": entity,
            }
        saves["entities"] = list(set(word_entities))
        saves["aliases"] = aliases
        saves["cooccurences"] = cooccurences
        saves["gpt"] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generated_content": generated_content,
            "params": params,
        }
        logger(saves)

    return find_cooccurences_aliases(cooccurences, aliases)


def get_cooccurences_with_aliases_and_gpt_NER(
    path: str, output_file_name: str = None
):
    """
    Get the aliases of the cooccurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """
    system_prompt = r"""
    Tu es un extracteur d'entités.
    Ton but est d'extraire tous les personnages du livre de science-fiction 'Le Cycle des Fondations' d'Isaac Asimov.
    Tu verras des passages du livre que tu devras utiliser. Dans ta définition, un personnage est un individu qui apparaît dans un passage du livre.
    Ce personnage peut être uniquement cité par son nom ou être très actif dans la discussion.
    Pour réaliser cette tâche, je souhaite que tu me retournes la liste des personnages que tu rencontres.
    Tu dois me donner l'ensemble des entités. Tu n'as pas le droit de modifier le nom des personnages ou d'en inventer de nouveaux, utilise seulement le texte.
    Tu peux avoir plusieurs références d'un même personnage, renvoie l'ensemble des références.
    Par exemple, Hari Seldon est souvent appelé Hari et/ou Seldon. Je veux que tu listes aussi cela.
    Le retour doit être fait dans un JSON.

    Voici des exemples :

    Exemple 1 :

    Texte :
    <start>
                            Mathématicien

    Output : C’est vraiment enfantin, dit @@Goutte-de-Pluie Quarante- cinq##. Nous pouvons vous montrer. — Nous allons vous préparer un bon repas bien nourrissant », dit @@Goutte-de-Pluie Quarante-trois##.

    Input : Mathématicien CLÉON Ier— ... dernier Empereur galactique de la
    dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
    année que Hari Seldon. (On pense que la date de naissance de
    Seldon, que certains estiment douteuse, aurait pu être
    « ajustée » pour coïncider avec celle de Cléon que Seldon, peu
    après son arrivée sur Trantor, est censé avoir rencontré.)

    Output :

    Input : Toutes les citations de l'Encyclopaedia Galactica reproduites ici
    proviennent de la 116e édition, publiée en 1020 E.F. par la Société
    d’édition de l'Encyclopaedia Galactica, Terminus, avec l'aimable
    autorisation des éditeurs.
    semblerait malgré tout qu’il puisse encore arriver des choses
    intéressantes. Du moins, à ce que j’ai entendu dire.
        — Par le ministre des Sciences ?
        — Effectivement. Il m’a appris que ce Hari Seldon a assisté
    à un congrès de mathématiciens ici même, à Trantor – ils
    l’organisent tous les dix ans, pour je ne sais quelle raison ; il
    aurait démontré qu’on peut prévoir mathématiquement

    Exemple 2 :

    Texte :
    <start>
        Étouffant un léger bâillement, Cléon demanda :
    « Demerzel, auriez-vous, par hasard, entendu parler d’un
    certain Hari Seldon ? »
        Cléon était empereur depuis dix ans à peine et, quand le
    protocole l’exigeait, il y avait des moments où, pourvu qu’il fût
    revêtu des atours et ornements idoines, il réussissait à paraître
    majestueux. Il y était arrivé, par exemple, pour son portrait
    <end>

    Output :

    {
    'personnages': ['Cléon', 'Demerzel', 'Hari Seldon', 'Cléon']
    }

    Exemple 3 :

    Texte :
    <start>
        2 Toutes les citations de l'Encyclopaedia Galactica reproduites ici
    proviennent de la 116e édition, publiée en 1020 E.F. par la Société
    d’édition de l'Encyclopaedia Galactica, Terminus, avec l'aimable
    autorisation des éditeurs.
    semblerait malgré tout qu’il puisse encore arriver des choses
    intéressantes. Du moins, à ce que j’ai entendu dire.
        — Par le ministre des Sciences ?
        — Effectivement. Il m’a appris que ce @@Hari Seldon## a assisté
    à un congrès de mathématiciens ici même, à Trantor – ils
    l’organisent tous les dix ans, pour je ne sais quelle raison ; il
    aurait démontré qu’on peut prévoir mathématiquement

    Output :

    Output : @@Seldon## savait qu’il n’avait pas le choix, nonobstant les circonlocutions polies de l’autre, mais rien ne lui interdisait de chercher à en savoir plus : « Pour voir @@l’Empereur##, @@Sire## ?

    Rappel : tu as interdiction d'inventer des personnages. Tu dois reprendre les noms des personnages dans le texte
    a l'identique. Tu peux avoir plusieurs références d'un même personnage, renvoie l'ensemble des références.
    Fait attention a ne pas considerer des personnages qui n'en sont pas comme des passants, un homme d'affaire, etc.

    Fait le pour l'exemple suivant.

    Texte :
    <start>
    """
    from vroom.alias import get_aliases_fuzzy
    from vroom.cooccurences import get_cooccurences
    from vroom.NER import read_file

    content = read_file(path)
    chunks = chunk_text_by_sentence(content, batch_size=5)

    entities = []

    client = OpenAI()

    for _, chunk in enumerate(chunks):
        print("chunk = ", chunk)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # gpt-4-1106-preview / gpt-3.5-turbo-1106
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": chunk + "\n<end>" + "\n\n Output : \n",
                },
            ],
            seed=42,
            temperature=0,
            #       presence_penalty=-2,
            response_format={"type": "json_object"},
        )

        generated_content = response.choices[0].message.content
        print()
        print(generated_content)
        print()
        print("content = ", generated_content)

        entities += generated_content["personnages"]
        print("total entities = ", set(entities))
        print("*" * 100)

        gpt_output = "*" * 100 + "\n"
        gpt_output += "<start>\n" + chunk + "\n<end>" + "\n\n Output : \n"
        gpt_output += str(set(entities)) + "\n"

    gpt_entities = set(entities)

    tagged_file = tag_text_with_entities(path, gpt_entities)
    out_entities = get_positions_of_entities(tagged_file)
    cooccurences = get_cooccurences([tagged_file], [out_entities])

    print("entities = ", entities)
    entities = [{"word": entity} for entity in entities]
    aliases = get_aliases_fuzzy(entities, 99)
    print("alises = ", aliases)

    return find_cooccurences_aliases(cooccurences, aliases)
