from clients.crossref import CrossRef
from clients.dandi import DandiClient
from clients.openai import OpenAIClient

from prompts import (
    STUDY_TARGET_SYSTEM_PROMPT,
    STUDY_TARGET_USER_PROMPT,
    KEYWORDS_SYSTEM_PROMPT,
    KEYWORDS_USER_PROMPT,
    STUDY_TARGET_AND_KEYWORD_SYSTEM_PROMPT,
    STUDY_TARGET_AND_KEYWORD_USER_PROMPT
)

from dandischema.models import Contributor
from keybert import KeyBERT

import concurrent.futures
import json


# num of keywords to be extracted from DOI
NUM_KEYWORDS = 10


class DOIExtraction:
    def __init__(self, doi: str, dandiset_id: str, keyword_extraction_type: str = "llm"):
        # Crossref
        self.doi = doi.strip()
        self.crossref_client = CrossRef()
        self.doi_metadata = self.crossref_client.get_doi_metadata(self.doi)
        if not self.doi_metadata:
            raise ValueError("Invalid DOI provided.")

        # Dandi
        self.dandiset_id = dandiset_id.strip()
        if len(dandiset_id.split("/")) != 2:
            raise ValueError("Invalid Dandiset ID provided.")
        self.dandi_client = DandiClient()
        id, version = self.dandiset_id.split("/")
        self.dandiset_metadata = self.dandi_client.get_raw_metadata(id, version)
        if not self.dandiset_metadata:
            raise ValueError("Invalid Dandiset ID provided.")
        
        self.dandiset_name = self.dandi_client.get_dandiset_name()
        self.dandiset_description = self.dandi_client.get_dandiset_description()

        # OpenAI
        self.openai_client = OpenAIClient()

        self.study_target = None
        self.keyword_extraction_type = keyword_extraction_type

    
    def get_contributors(self):
        """Return a list of dandischema.models.Contributor"""
        contributors = self.crossref_client.get_contributors()
        if not contributors:
            return None
        
        dandischema_contributors = []
        with concurrent.futures.ThreadPoolExecutor() as exec:
            futures = []
            for contributor in contributors:
                futures.append(exec.submit(self._process_contributor, contributor))

            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    dandischema_contributors.append(future.result())

        return dandischema_contributors
    

    def _process_contributor(self, contributor):
        affiliations = self.crossref_client.get_author_affiliations(contributor)
        
        if any(key in contributor for key in ["ORCID", "family", "given"]):
            family_name, given_name = self.crossref_client.get_author_full_name(contributor)
            if not family_name or not given_name:
                return None
            else:
                name = f"{family_name.strip().title()}, {given_name.strip().title()}"

            orcid = self.crossref_client.get_author_orcid(contributor)
            if orcid:
                orcid = orcid.split("/")[-1]
                email, url = self.crossref_client.get_info_from_orcid(orcid)
            else:
                email, url = None, None
                
            # person
            schema = self.dandi_client.filled_person_schema(
                orcid=orcid,
                name=name,
                email=email,
                url=url,
                affiliation=affiliations,
            )
        else:
            # organization
            schema = self.dandi_client.filled_organization_schema(
                ror=None,
                name=contributor.get("name", None),
                email=None,
                url=None,
                role=None,
            )
        return schema
    

    # NOTE: testing approach to combine study target and keywords
    def get_study_target_and_keywords(self) -> (str, list[str]):
        subjects = self.crossref_client.get_subjects()

        doi_title = self.crossref_client.get_title()
        doi_abstract = self.crossref_client.get_abstract()

        if subjects and len(subjects) > 0:
            expert_prompt = f"(specicially in {', '.join(subjects)})"
        else:
            expert_prompt = ""

        system_prompt = STUDY_TARGET_AND_KEYWORD_SYSTEM_PROMPT.format(expert_prompt)
        user_prompt = STUDY_TARGET_AND_KEYWORD_USER_PROMPT.format(
            NUM_KEYWORDS, NUM_KEYWORDS,
            "NA" if not self.dandiset_name else self.dandiset_name,
            "NA" if not self.dandiset_description else self.dandiset_description,
            "NA" if not doi_title else doi_title,
            "NA" if not doi_abstract else doi_abstract
        )

        response = self.openai_client.get_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=250
        )
        json_response = json.loads(response)
        study_target = json_response["study_target"]
        keywords = json_response["keywords"]
        return study_target, keywords


    def get_study_target(self):
        subjects = self.crossref_client.get_subjects()
        
        doi_title = self.crossref_client.get_title()
        doi_abstract = self.crossref_client.get_abstract()

        if subjects and len(subjects) > 0:
            expert_prompt = f"You are an expert in the following subjects: {', '.join(subjects)}"
        else:
            expert_prompt = ""

        user_prompt = STUDY_TARGET_USER_PROMPT.format(
            expert_prompt,
            "None" if not self.dandiset_name else self.dandiset_name,
            "None" if not self.dandiset_description else self.dandiset_description,
            "None" if not doi_title else doi_title,
            "None" if not doi_abstract else doi_abstract
        )

        MAX_TOKENS = 100
        self.study_target = self.openai_client.get_llm_response(
            system_prompt=STUDY_TARGET_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=MAX_TOKENS
        )
        return self.study_target
    

    def get_keywords(self, study_target_test: str = None, type="llm"):
        if study_target_test:
            study_target = study_target_test
        else:
            study_target = self.study_target if self.study_target else self.get_study_target()

        if type == "keybert":
            keywords = self._get_keywords_keybert(study_target)
        elif type == "llm":
            keywords = self._get_keywords_llm(study_target)
        else:
            print("Invalid keyword extraction type.")
            return None

        return keywords
    

    def _get_keywords_keybert(self, study_target: str):
        kw_model = KeyBERT()

        stop_words = None

        keywords = kw_model.extract_keywords(
            study_target,
            keyphrase_ngram_range=(1, 2),
            top_n=NUM_KEYWORDS,
            diversity=0.7,
            stop_words=stop_words,
        )
        return [kw[0] for kw in keywords]


    def _get_keywords_llm(self, study_target: str):
        user_prompt = KEYWORDS_USER_PROMPT.format(
            study_target,
            NUM_KEYWORDS,
            NUM_KEYWORDS
        )

        choices = self.openai_client.get_llm_response(
            system_prompt=KEYWORDS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        keywords = choices.split(", ")
        return keywords
    
    
    def __str__(self):
        contributors: list[Contributor] = self.get_contributors()

        # use combined LLM approach for study target generation and keyword extraction
        combined = True
        if combined:
            study_target, keywords = self.get_study_target_and_keywords()
        else:
            study_target: list[str] = [self.get_study_target()]
            keywords: list[str] = self.get_keywords(type=self.keyword_extraction_type)
        
        return str({
            "contributors": contributors,
            "study_target": [study_target],
            "keywords": keywords
        })
        