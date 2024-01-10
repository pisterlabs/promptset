import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from text import nonewlines

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class RetrieveThenReadApproach(Approach):

    template = \
"You are an intelligent assistant helping users with health issues by answering their questions based on healthcare knowledge base documentation. " \
"Use 'you' to refer to the individual asking the questions even if they ask with 'I'. " + \
"Answer the question using only the data provided in the information sources below. " + \
"For tabular information return it as an html table. Do not return markdown format. " + \
"Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. " + \
"If you cannot answer using the sources below, say you don't know. You answer in the language of the user." + \
"""

###
Question: 'Kerro minulle ALS taudista?'

Sources:
linked_page_7735_12.pdf: ALS on harvinainen, etenevä hermoston sairaus, joka vaikuttaa lihasten toimintaan.

Answer:
ALS (amyotrofinen lateraaliskleroosi) on harvinainen, etenevä hermoston sairaus, joka vaikuttaa lihasten toimintaan. ALS-tauti vaurioittaa motorisia hermosoluja, jotka ohjaavat tahdonalaisia lihaksia, ja tätä kautta heikentää lihasten voimaa ja toimintakykyä.

ALS-taudin oireita ovat:

Lihasten heikkous ja surkastuminen
Lihasten nykiminen (faskikulaatiot)
Lihasten jäykkyys (spastisuus)
Puheen, nielemisen ja hengityksen vaikeudet

ALS-taudin syyt ovat pääosin tuntemattomia, mutta taudin kehittymisessä saattaa olla geneettisiä tekijöitä. Tauti alkaa yleensä iän 40 ja 70 välillä, ja miesten sairastumisriski on hieman suurempi kuin naisten.

Tällä hetkellä ei ole parantavaa hoitoa ALS-taudille, mutta oireita voidaan lievittää ja elämänlaatua parantaa esimerkiksi fysio- ja puheterapian avulla. Lääkityksellä voidaan hidastaa taudin etenemistä ja helpottaa oireita.

###
Question: '{q}'?

Sources:
{retrieved}

Answer:
"""

    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, q: str, overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="fi-FI", 
                                          # query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        prompt = (overrides.get("prompt_template") or self.template).format(q=q, retrieved=content)
        completion = openai.Completion.create(
            engine=self.openai_deployment, 
            prompt=prompt, 
            temperature=overrides.get("temperature") or 0.3, 
            max_tokens=1024, 
            n=1, 
            stop=["\n"])

        return {"data_points": results, "answer": completion.choices[0].text, "thoughts": f"Question:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
