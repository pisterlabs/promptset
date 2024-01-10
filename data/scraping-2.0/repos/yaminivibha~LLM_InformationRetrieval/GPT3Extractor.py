"GPT3 Extractor class"
import json
import re
from typing import List, Set, Tuple

import openai
import spacy
from spacy_help_functions import create_entity_pairs

from lib.utils import (
    ENTITIES_OF_INTEREST,
    PROMPT_AIDS,
    PRONOUNS_AND_CONJUNCTIONS,
    RELATIONS,
    SEED_PROMPTS,
    SEED_SENTENCES,
    SUBJ_OBJ_REQUIRED_ENTITIES,
)


class gpt3Extractor:
    """
    GPT3 Extractor class
    """

    def __init__(self, r, openai_key, model="en_core_web_sm"):
        """
        Initialize a gpt3Predictor object
        Parameters:
            r: the relation to extract
            openai_key: the key to use for the OpenAI API
            model: the spaCy model to use
        """
        self.openai_key = openai_key
        openai.api_key = self.openai_key
        self.nlp = spacy.load(model)
        self.r = r
        self.relations = set()

    def get_relations(self, text: str) -> List[Tuple[str, str]]:
        """
        Exposed function to take in text and return named entities
        Parameters:
            text: the text to extract entities from
        Returns:
            entities: a list of tuples of the form (subject, object)
        """
        doc = self.nlp(text)
        print("        Annotating the webpage using spacy...")
        num_sents = len(list(doc.sents))
        print(
            f"        Extracted {num_sents} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ..."
        )

        # Get tagged version of text from spaCy.
        target_candidate_pairs = self.extract_candidate_pairs(doc)

        if len(target_candidate_pairs) == 0:
            return []
        # print("target_candidate_pairs: {}".format(target_candidate_pairs))
        self.extract_entity_relations(target_candidate_pairs)
        return self.relations

    def extract_candidate_pairs(self, doc) -> Set[Tuple[str, str]]:
        """
        Extract candidate pairs from a given document using spaCy
        parameters:
            doc: the document to extract candidate pairs from
        returns:
            relations: a list of candidate entity pairs, where each item is a tuple
                                    (subj, obj)
        """
        num_sents = len(list(doc.sents))
        extracted_sentences = 0
        extracted_annotations = 0
        for i, sentence in enumerate(doc.sents):
            if i % 5 == 0 and i != 0:
                print(f"        Processed {i} / {num_sents} sentences")
            # Create entity pairs
            sentence_entity_pairs = create_entity_pairs(
                sentence, ENTITIES_OF_INTEREST[self.r]
            )

            # Check entity pairs if any appropriate subj/obj pairing exists
            candidates = self.filter_candidates_exist(sentence_entity_pairs)

            # If any viable candidates exist, pass to GPT-3 for extraction
            if candidates:
                relation = self.extract_entity_relations(sentence)
                output = self.parse_gpt_output(relation)
                # If GPT-3 returns invalid relation, move on to next sentence
                if not output:
                    continue
                # If GPT-3 returns valid relation, check if it's a duplicate
                output_tuple = (output["subj"], output["obj"])
                if output_tuple not in self.relations:
                    # If not a duplicate, add to set, print output
                    self.relations.add(output_tuple)
                    extracted_annotations += 1
                    extracted_sentences += 1
                    self.print_output_relation(sentence, output, duplicate=False)
                else:
                    # If duplicate, print output and move on
                    self.print_output_relation(sentence, output, duplicate=True)

        print(
            f"Extracted annotations for  {extracted_sentences}  out of total  {num_sents}  sentences"
        )
        print(
            f"Relations extracted from this website: {extracted_annotations} (Overall: {len(self.relations)})"
        )
        return self.relations

    def print_output_relation(self, sentence, output, duplicate):
        print("                === Extracted Relation ===")
        print(f"                Sentence:  {sentence}")
        print(f"                Subject: {output['subj']} ; Object: {output['obj']} ;")
        if duplicate:
            print("                Duplicate. Ignoring this.")
        else:
            print("                Adding to set of extracted relations")
        print("                ==========")

    def filter_candidates_exist(self, sentence_entity_pairs: List) -> bool:
        """
        Filter candidate pairs to only include those that are of the right type
        Parameters:
            sentence_entity_pairs: a list of candidate entity pairs, where each pair is a dictionary
        Returns:
            bool: if at least 1 viable candidate pair exists, return True. Else, False.
        """
        # Create candidate pairs. Filter out subject-object pairs that
        # aren't the right type for the target relation.
        # (e.g. don't include anything that's not Person:Organization for the "Work_For" relation)
        candidate_pairs = []
        for ep in sentence_entity_pairs:
            candidate_pairs.append(
                {"tokens": ep[0], "subj": ep[1], "obj": ep[2]}
            )  # e1=Subject, e2=Object
            candidate_pairs.append(
                {"tokens": ep[0], "subj": ep[2], "obj": ep[1]}
            )  # e1=Object, e2=Subject

        for p in candidate_pairs:
            if (
                p["subj"][1] in SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["SUBJ"]
                and p["obj"][1] in SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["OBJ"]
            ):
                return True

        print("		No potential relations found in this sentence...")
        # This info, formatted, should be printed in extract_candidate_pairs.
        # print("Filtered target_candidate_paris: {}".format(target_candidate_pairs))
        return False

    def parse_gpt_output(self, output_str: str):
        """
        Parse the output of GPT-3
        Parameters:
            output: the output of GPT-3, string '{"PERSON": "John Doe", "ORGANIZATION": "Google", "RELATION": "Work_For"}'
        Returns:
            resultant_relation: the extracted relation as a dict
                        with format:
                        {
                            "subj": <subject>,
                            "obj": <object>,
                            "relation": <relation>
                        }
            If any KeyError in the GPT output, return None
        Raises:
            None
        """
        resultant_relation = {}
        try:
            output = json.loads(output_str)
            resultant_relation["subj"] = output[
                SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["SUBJ"][0].strip()
            ]
            resultant_relation["obj"] = output[
                SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["OBJ"][0].strip()
            ]
            resultant_relation["relation"] = output["RELATION"]

            # This filters out any relations that don't match the target relation.
            # It also filters out blank or "n/a" subject and objects.
            # It also filters out relations where subject is/contains a prounoun
            if resultant_relation["relation"] != RELATIONS[self.r]:
                resultant_relation = None
            if resultant_relation["subj"] == "" or resultant_relation["obj"] == "":
                resultant_relation = None
            if (
                resultant_relation["subj"] == "None"
                or resultant_relation["obj"] == "None"
            ):
                resultant_relation = None
            if (
                resultant_relation["subj"] == "n/a"
                or resultant_relation["obj"] == "n/a"
            ):
                resultant_relation = None
            if (
                resultant_relation["subj"] == "N/A"
                or resultant_relation["obj"] == "N/A"
            ):
                resultant_relation = None
            if any(
                p in resultant_relation["subj"].lower()
                for p in PRONOUNS_AND_CONJUNCTIONS
            ):
                resultant_relation = None
        except Exception:
            print(f"Error parsing GPT-3 output: {output_str}")
            resultant_relation = None
        return resultant_relation

    def extract_entity_relations(self, sentence):
        """
        Extract entity relations
        Parameters:
            candidate_pairs: a list of candidate pairs to extract relations from
        Returns:
            relations: a list of tuples of the form (subject, object)
        """
        prompt = self.construct_prompt(sentence)
        relation = self.gpt3_complete(prompt)
        return relation

    def gpt3_complete(self, prompt):
        """
        Use GPT-3 to complete a prompt
        Parameters:
            prompt: the prompt to complete
        Returns:
            completion: the completion of the prompt
        """
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.2,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return completion["choices"][0]["text"]

    def construct_prompt(self, sentence):
        """
        Construct a prompt for GPT-3 to complete.
        Parameters:
            candidate_pairs: a single candidate pairs to extract relations from
        Returns:
            prompt: a string to be passed to GPT-3
        """
        seed = f"In a given sentence, find relations where {PROMPT_AIDS[self.r]}"
        example = f"Example Input: '{SEED_SENTENCES[self.r]}' Example Output: {SEED_PROMPTS[self.r]}."
        sentence = f"Input: {sentence} Output:"

        return seed + example + sentence
