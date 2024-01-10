import os
import openai
import json

from util.logger import get_logger
from util.lang_chain_util import create_llm, create_llm_chain
from typing import List, Dict

from langchain.prompts import PromptTemplate

openai.api_key = os.environ.get("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")


class DoubleCheckService:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__, config)

    # TODO: add logging
    def double_check(self, analyzed_situation_with_law: List[Dict]) -> List[Dict]:
        prompt_list = []
        analyzed_situation_with_law_and_double_checked = []
        for situation in analyzed_situation_with_law:
            for aktion in situation["aktionen"]:
                prompt_list.append(self.evaluate_prompt(aktion))
            count = 1

            for prompt in prompt_list:
                aktion = self.get_most_relevant_article(prompt, count, situation)
                count += 1
            analyzed_situation_with_law_and_double_checked.append(situation)
        return analyzed_situation_with_law_and_double_checked

    def get_most_relevant_article(
        self, prompt: str, count: int, situation: dict
    ) -> Dict:
        doublecheck_prompt = PromptTemplate(
            input_variables=["situation_and_law"],
            template="""\
            Du bekommst die folgendenn Text mit einem Sacherhalt und Gesetzesartikeln {situation_and_law} mit der Artikel-Referenz und den Artikeltext. Prüfe welcher der Gesetzesartikel am besten zum Sachverhalt passt und liefere nur den diesen Artikel zurück und gib ein Objekt in folgender Struktur zurück wie folgt, ergänze keine Attribute:
            "beschreibung": "Text von Sachverhalt", "artikel": Artikel als key:value pair, wobei der key der Artikel ist und der value der Text.
                
            Formatiere die Antwort in ein JSON.""",
        )
        chain = create_llm_chain(create_llm(), doublecheck_prompt)
        output = chain.run({"situation_and_law": prompt})
        dict_output = json.loads(output)
        for aktion in situation["aktionen"]:
            if aktion["id"] == count:
                aktion["artikel"] = dict_output["artikel"]
        return aktion

    def evaluate_prompt(self, aktion: dict) -> str:
        artikel_infos = " ".join(
            [f"{artikel}: {text}" for artikel, text in aktion["artikel"].items()]
        )
        return f"Sachverhalt: {aktion['beschreibung']}\nArtikel-Infos: {artikel_infos}"
