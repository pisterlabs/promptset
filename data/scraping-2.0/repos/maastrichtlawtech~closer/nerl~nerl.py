import csv
import os
import re

import nltk
import spacy
import tagme
from nltk.stem.snowball import SnowballStemmer
from refined.inference.processor import Refined

from legal_openai.openai_tasks import OpenaiTask


class EntityRecognizer:
    def __init__(self):
        self.prompt_path = os.environ.get('PROMPT_PATH')

    def spacy_recognize(self, text, model_path='en_core_web_lg'):
        try:
            self.nlp = spacy.load(model_path)
            nltk.download('punkt')
        except OSError:
            spacy.cli.download(model_path)
            self.nlp = spacy.load(model_path)

        self.nlp.add_pipe("entityLinker", last=True)
        doc = self.nlp(text)
        entity_uri = {}
        for index, entity_element in enumerate(doc._.linkedEntities):
            entity_uri[entity_element.label] = entity_element.url
        return entity_uri

    def tsv_to_dict(self, tsv_file):
        eurovoc_dict = {}
        eurovoc_reverse_dict = {}
        uri_list = []
        concept_list = []
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            row_count = 0
            for row in reader:
                row_count += 1
                col_count = 0
                if row_count > 1:
                    for cells in row:
                        col_count += 1
                        if col_count == 1:
                            uri_list.append(cells)
                            key = cells
                        else:
                            concept_list.append(cells)
                            value = cells
                    eurovoc_dict[key] = value
                    eurovoc_reverse_dict[value] = key
        return eurovoc_dict, eurovoc_reverse_dict, uri_list, concept_list

    def token_cleaning(self, token, stemmer):
        token = token.lower()
        token = stemmer.stem(token)
        return token

    def RegexFromTerm(self, term, stemmer, regex=r"\b("):
        # Adding terms to regex
        tokens_list = nltk.word_tokenize(term)
        if len(tokens_list) == 1:
            for token in tokens_list:
                if token != '':
                    regex += self.token_cleaning(token, stemmer)
        else:
            count = len(tokens_list)
            for token in tokens_list:
                count = count - 1
                if count != len(tokens_list) - 1:
                    regex += r'\w*\W\w*\W*'
                regex += self.token_cleaning(token, stemmer)
        # Regex closure 
        regex += '''\w{0,5})(\W)'''
        return regex

    def eurovoc_recognize(self, text, tsv_file=None,
                          eurovoc_link='http://eurovoc.europa.eu/'):
        eurovoc_dict, eurovoc_reverse_dict, uri_list, concept_list = self.tsv_to_dict( \
            tsv_file)
        stemmer_en = SnowballStemmer("english")
        text = text.lower()
        entity_uri = {}
        for concept in concept_list:
            if concept != '':
                regex = self.RegexFromTerm(concept, stemmer_en)
                if re.search(regex, text) is not None:
                    entity_uri[concept] = eurovoc_link + eurovoc_reverse_dict[concept]
        return entity_uri

    def wikipedia_tagme(self, text, tagme_api_key, threshold=0.5):
        tagme.GCUBE_TOKEN = tagme_api_key
        annotations = tagme.annotate(text)
        entity_uri = {}
        for ann in annotations.get_annotations(threshold):
            entity_uri[ann.entity_title] = ann.uri()
        return entity_uri

    def refined_recognize(self, text, model_name='wikipedia_model',
                          entity_set='wikidata',
                          base_uri='http://www.wikidata.org/wiki/'):
        refined = Refined.from_pretrained(model_name=model_name, entity_set=entity_set)
        spans = refined.process_text(text)
        entity_uri = {}
        for ent in spans:
            entity_uri[ent.text] = base_uri + str(ent.predicted_entity.wikidata_entity_id)
        return entity_uri

    def openai_recognize(self, api_key=None, article=None, prompt=None, path=None,
                         use_index=True):
        if prompt is None:
            with open(self.prompt_path + '/normal_prompts/entity_recognition.txt', 'r') as f:
                prompt = f.read()
        response = OpenaiTask(path=path, api_key=api_key, use_index=use_index).execute_task(
            article=article, prompt=prompt)
        return response

    def openai_wikidata_recognize(self, api_key=None, article=None, prompt=None,
                                  path=None, use_index=True):
        if prompt is None:
            with open(self.prompt_path + '/normal_prompts/wikidata_entities.txt', 'r') as f:
                prompt = f.read()
        response = OpenaiTask(path=path, api_key=api_key, use_index=use_index).execute_task(
            article=article, prompt=prompt)
        return response

    def openai_eurovoc_recognize(self, api_key=None, article=None, prompt=None,
                                 path=None, use_index=True):
        if prompt is None:
            with open(self.prompt_path + '/normal_prompts/eurovoc_recognition.txt', 'r') as f:
                prompt = f.read()
        response = OpenaiTask(path=path, api_key=api_key, use_index=use_index).execute_task(
            article=article, prompt=prompt)
        return response
