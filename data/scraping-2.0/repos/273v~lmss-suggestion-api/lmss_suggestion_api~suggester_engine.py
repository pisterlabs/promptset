"""
Suggester engine to provide suggestions from a given text input through the API or CLI.
"""

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# imports
import json
import logging
from enum import Enum

# packages
import openai
import rapidfuzz.fuzz
import tiktoken
import telly
import sys
from lmss.graph import LMSSGraph

# project imports
from lmss_suggestion_api.api_settings import ENV


# set the openai key
if ENV.get("OPENAI_API_KEY", None) in [None, ""]:
    LLM_ENABLED = False
else:
    LLM_ENABLED = True
    openai.api_key = ENV["OPENAI_API_KEY"]


class SuggestionType(Enum):
    """SuggestionType enum provides the supported types of suggestions."""

    ACTOR_PLAYER = "Actor / Player"
    AREA_OF_LAW = "Area of Law"
    ASSET_TYPE = "Asset Type"
    COMMUNICATION_MODALITY = "Communication Modality"
    CURRENCY = "Currency"
    DATA_FORMAT = "Data Format"
    DOCUMENT_ARTIFACT = "Document / Artifact"
    ENGAGEMENT_TERMS = "Engagement Terms"
    EVENT = "Event"
    FORUM_AND_VENUE = "Forums and Venues"
    GOVERNMENTAL_BODY = "Governmental Body"
    INDUSTRY = "Industry"
    LMSS_TYPE = "LMSS Type"
    LEGAL_AUTHORITIES = "Legal Authorities"
    LEGAL_ENTITY = "Legal Entity"
    LOCATION = "Location"
    MATTER_NARRATIVE = "Matter Narrative"
    MATTER_NARRATIVE_FORMAT = "Matter Narrative Format"
    OBJECTIVE = "Objectives"
    SERVICE = "Service"
    STANDARDS_COMPATIBILITY = "Standards Compatibility"
    STATUS = "Status"
    SYSTEM_IDENTIFIERS = "System Identifiers"


# mapping from concept name/label to router endpoint
CONCEPT_TO_ENDPOINT = {
    "Actor / Player": "/suggest/actor-player",
    "Area of Law": "/suggest/area-of-law",
    "Asset Type": "/suggest/asset-type",
    "Communication Modality": "/suggest/communication-modality",
    "Currency": "/suggest/currency",
    "Data Format": "/suggest/data-format",
    "Document / Artifact": "/suggest/document-artifact",
    "Engagement Terms": "/suggest/engagement-terms",
    "Event": "/suggest/event",
    "Forums and Venues": "/suggest/forum-and-venue",
    "Governmental Body": "/suggest/governmental-body",
    "Industry": "/suggest/industry",
    "LMSS Type": "/suggest/lmss-type",
    "Legal Authorities": "/suggest/legal-authorities",
    "Legal Entity": "/suggest/legal-entity",
    "Location": "/suggest/location",
    "Matter Narrative": "/suggest/matter-narrative",
    "Matter Narrative Format": "/suggest/matter-narrative-format",
    "Objectives": "/suggest/objective",
    "Service": "/suggest/service",
    "Standards Compatibility": "/suggest/standards-compatibility",
    "Status": "/suggest/status",
    "System Identifiers": "/suggest/system-identifiers",
}


# pylint: disable=R0902,R0913
class SuggesterEngine:
    """SuggesterEngine class provides an implementation of the multi-stage suggestion engine
    used to serve the SALI suggester API.  The suggester engine attempts to assign one or more
    labels to the input text based on the following:
      - [ ] Exact matchings of the input text to the labels in the ontology
      - [ ] Fuzzy matching via LSH(F) of the input text to the labels in the ontology
      - [ ] Word/sentence embeddings similarity of the input text to the labels in the ontology
      - [ ] LLM (e.g., GPT, T5, KLLM) based suggestions
    """

    def __init__(
        self,
        enable_label_search: bool = True,
        enable_definition_search: bool = True,
        enable_llm_search: bool = True,
        llm_distance_threshold: float = 0.2,
        max_llm_concept_tokens: int = 2048,
        include_hidden_labels: bool = True,
        include_alt_labels: bool = True,
        max_distance: float = 0.2,
        max_results: int = 10,
        logger: logging.Logger | None = None,
    ):
        """Constructor for SuggesterEngine class.

        Args:
            enable_label_search (bool, optional): Enable label search. Defaults to True.
            enable_definition_search (bool, optional): Enable definition search. Defaults to True.
            enable_llm_search (bool, optional): Enable LLM search. Defaults to True.
            llm_distance_threshold (float, optional): LLM distance threshold. Defaults to 0.2.
            max_llm_concept_tokens (int, optional): Maximum number of tokens to use for LLM
                suggestions. Defaults to 2048.
            include_hidden_labels (bool, optional): Include hidden labels in suggestions. Defaults to True.
            include_alt_labels (bool, optional): Include alternative labels in suggestions. Defaults to True.
            max_distance (int, optional): Maximum edit distance to consider for fuzzy matching. Defaults to 50.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.
        """
        # set stage enablement
        self.enable_label_search = enable_label_search
        self.enable_definition_search = enable_definition_search
        self.enable_llm_search = enable_llm_search
        self.llm_distance_threshold = llm_distance_threshold
        self.max_llm_concept_tokens = max_llm_concept_tokens
        self.include_hidden_labels = include_hidden_labels
        self.include_alt_labels = include_alt_labels
        self.min_llm_char_length = 6
        self.max_llm_char_length = 500
        self.max_distance = max_distance
        self.max_results = max_results

        # setup logger
        self.logger = logger or logging.getLogger(__name__)

        # initialize LMSS ontology structure from python lmss package
        self.lmss: LMSSGraph = LMSSGraph(owl_branch=ENV["LMSS_BRANCH"])

        # setup tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(ENV["OPENAI_MODEL_NAME"])
        except Exception as ex:  # pylint: disable=W0703
            self.logger.warning(
                f"Error loading tokenizer: {ex}; setting default tokenizer"
            )
            self.tokenizer = tiktoken.encoding_for_model("text-davinci-003")

    def get_llm_meta_prompt_text(
        self,
        concept_list: list[SuggestionType] | None = None,
    ) -> str:
        """Get a query designed to be used as a meta prompt for the LLM, which asks the LLM
        to suggest top level/key concepts based on the input query.

        Args:
            concept_list (list[SuggestionType], optional): List of concepts to include in the prompt. Defaults to None.

        Returns:
            str: The meta prompt text.
        """

        # get the concept list
        if concept_list is None:
            # include all suggestion types
            concept_list = list(SuggestionType)

        # get the list of concept labels
        prompt = "<CONCEPT TYPES>\n"
        for concept in concept_list:
            concept_data = self.lmss.concepts[self.lmss.key_concepts[concept.value]]
            if len(concept_data["definitions"]) > 0:
                definition = concept_data["definitions"][0]
                prompt += f" - {concept.value}: {definition}\n"
            else:
                prompt += f" - {concept.value}\n"

        return prompt

        # pylint: disable=W0718

    async def get_llm_meta_response(
        self,
        query: str,
        model: str = ENV["OPENAI_MODEL_NAME"],
        max_retry: int = 1,
    ) -> dict:
        """Get a response from the LLM regarding the meta-query for concept types to
        use for the LLM.

        Args:
            query (str): Query to use for the LLM.
            model (str): Model to use for the LLM.
            max_retry (int): Maximum number of retries to make if the API fails.

        Returns:
            dict: Response from the LLM API.
        """

        response_data = None
        retry_count = 0

        while response_data is None and retry_count < max_retry:
            if retry_count == max_retry:
                raise RuntimeError("Too many retries.")

            contextual_prompt = """<PROMPT>You are an experienced paralegal working in the back office of a 
law firm or corporate legal department.  You review user input and provide suggestions for the best way to organize
information using the LMSS ontology.  

Perform the following steps to complete the task:

1. Review the list of top level concepts provided in CONCEPT TYPES section above.  

2. Review the user input provided in INPUT section.

3. Select zero or more concepts from the list that best describe the user input.  

4. Rank the selected concepts in order of relevance to the user input.

5. Return the ordered labels as a JSON array of label,score pairs like [["label", 0.877], ...]

6. Only respond in JSON.  Do not include any other text in your response."""

            try:
                # switch on model types
                if "turbo" in model.lower():
                    system_prompt = self.get_llm_meta_prompt_text()
                    system_prompt += contextual_prompt

                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": f"""<INPUT>{query}"""},
                    ]

                    # calculate token length
                    token_count = sum(
                        len(self.tokenizer.encode(m["content"])) for m in messages
                    )

                    response = await openai.ChatCompletion.acreate(
                        model=model,
                        max_tokens=4000 - token_count,
                        temperature=0.0,
                        messages=messages,
                    )

                    # get the last choices/[]/message/content response
                    if len(response["choices"]) > 0:
                        try:
                            response_data = json.loads(
                                response["choices"][-1]["message"]["content"]
                            )
                            return response_data
                        except Exception as error:
                            self.logger.error(
                                "Error parsing response from OpenAI ChatCompletion API: %s \ndata=%s",
                                error,
                                response,
                            )
                            retry_count += 1
                            continue

                    response_data = None
                else:
                    prompt = self.get_llm_meta_prompt_text()
                    prompt += contextual_prompt + "\n"
                    prompt += f"""<INPUT>{query}"""

                    # calculate token length
                    token_count = len(self.tokenizer.encode(prompt))

                    response = await openai.Completion.acreate(
                        model=model,
                        prompt=prompt,
                        max_tokens=4000 - token_count,
                        temperature=0.0,
                    )
                    # set response if present
                    if len(response["choices"]) > 0:
                        try:
                            response_data = json.loads(response["choices"][0]["text"])
                            return response_data
                        except Exception as error:
                            self.logger.error(
                                "Error parsing response from OpenAI Completion API: %s \ndata=%s",
                                error,
                                response,
                            )
                            retry_count += 1
                            continue
                    else:
                        response_data = None
            except Exception as error:
                # log
                self.logger.error(
                    "Error getting response from OpenAI API: %s",
                    error,
                )
                retry_count += 1
                continue

        # return
        if not response_data:
            return {}
        return response_data

    def get_llm_prompt_text(
        self,
        query: str,
        suggestion_type: SuggestionType,
        max_depth: int = 3,
        max_length: int = 2048,
    ) -> str:
        """Get the prompt to use for the LLM.
        Args:
            query (str): Query to use for the LLM.
            suggestion_type (SuggestionType): Suggestion type to use for the LLM.
            max_depth (int, optional): Maximum depth to use for the LLM. Defaults to 3.
            max_length (int, optional): Maximum length to use for the LLM. Defaults to 1024.
        Returns:
            str: Prompt to use for the LLM.
        """
        # initialize the return value
        # get concept labels as a list
        concept_labels = [
            self.lmss.concepts[iri]["label"]
            for iri in self.lmss.get_children(
                self.lmss.key_concepts[suggestion_type.value]
            )
            if self.lmss.concepts[iri]["label"] not in [None, ""]
        ]
        concept_label_list = "; ".join(concept_labels)
        concept_token_count = len(self.tokenizer.encode(concept_label_list))
        if concept_token_count > max_length:
            return self.get_llm_prompt_text(
                query,
                suggestion_type=suggestion_type,
                max_depth=max_depth - 1,
                max_length=max_length,
            )

        prompt = f"""<LABELS>
{concept_label_list}
<TEXT>
{query}
<PROMPT>You are a legal professional working in the back office of a law firm or corporate legal department.  Follow these instructions:
1. Read the TEXT above.
2. Read the {SuggestionType.value} LABELS above.
3. Assign zero or more labels from the LABELS list to the TEXT.
4. Rank order the labels in the order of relevance to the TEXT.
5. Return the ordered labels as a JSON array of label,score pairs like [["label", 0.877], ...]
6. Only respond in JSON.  Do not include any other text in your response.
<OUTPUT>"""

        # return
        return prompt

    def get_llm_prompt_chat(
        self,
        query: str,
        suggestion_type: SuggestionType,
        max_depth: int = 8,
        max_length: int = 2048,
    ) -> str:
        """Get the prompt to use for the LLM.

        Args:
            query (str): Query to use for the LLM.
            suggestion_type (SuggestionType): Suggestion type to use for the LLM.
            max_depth (int, optional): Maximum depth to use for the LLM. Defaults to 8.
            max_length (int, optional): Maximum length to use for the LLM. Defaults to 2000.

        Returns:
            str: Prompt to use for the LLM.
        """
        # initialize the return value
        # get concept labels as a list
        concept_labels = [
            self.lmss.concepts[iri]["label"]
            for iri in self.lmss.get_children(
                self.lmss.key_concepts[suggestion_type.value],
                max_depth=max_depth,
            )
            if self.lmss.concepts[iri]["label"] not in [None, ""]
        ]
        concept_label_list = "; ".join(concept_labels)
        concept_token_count = len(self.tokenizer.encode(concept_label_list))
        if concept_token_count > max_length:
            return self.get_llm_prompt_chat(
                query,
                suggestion_type=suggestion_type,
                max_depth=max_depth - 1,
                max_length=max_length,
            )

        prompt = f"""<ROLE>
You are a legal professional working in the back office of a law firm or corporate legal department.
</ROLE>

<LABELS>
{concept_label_list}
</LABELS>

<PROMPT>Follow these instructions:
1. Assign zero or more of the LABELS above to the user input.
2. Rank order the labels in order of relevance to the user input.
3. Return the ordered labels as a JSON array of label,score pairs like [["label", 0.877], ...]
4. Respond only in JSON.  If you do not understand the user input, return an empty JSON array.
</PROMPT>"""

        # return
        return prompt

    # pylint: disable=W0718
    async def get_llm_response(
        self,
        query: str,
        suggestion_type: SuggestionType,
        model: str = ENV["OPENAI_MODEL_NAME"],
        max_retry: int = 1,
    ) -> dict:
        """Get the LLM API response for the prompt; only using OpenAI for now.

        Args:
            query (str): Query to use for the LLM.
            suggestion_type (SuggestionType): Suggestion type to use for the LLM.
            model (str): Model to use for the LLM.
            max_retry (int): Maximum number of retries to make if the API fails.

        Returns:
            dict: Response from the LLM API.
        """

        response_data = None
        retry_count = 0

        while response_data is None and retry_count < max_retry:
            if retry_count == max_retry:
                raise RuntimeError("Too many retries.")

            try:
                # switch on model types
                if "turbo" in model.lower():
                    messages = [
                        {
                            "role": "system",
                            "content": self.get_llm_prompt_chat(
                                query=query,
                                suggestion_type=suggestion_type,
                                max_length=self.max_llm_concept_tokens,
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"<TEXT>{query}</TEXT>",
                        },
                    ]

                    # calculate token length
                    token_count = sum(
                        len(self.tokenizer.encode(m["content"])) for m in messages
                    )

                    response = await openai.ChatCompletion.acreate(
                        model=model,
                        max_tokens=4000 - token_count,
                        temperature=0.0,
                        messages=messages,
                    )

                    # get the last choices/[]/message/content response
                    if len(response["choices"]) > 0:
                        try:
                            response_data = json.loads(
                                response["choices"][-1]["message"]["content"]
                            )
                            return response_data
                        except Exception as error:
                            self.logger.error(
                                "Error parsing response from OpenAI ChatCompletion API: %s\ndata=%s",
                                error,
                                response,
                            )
                            retry_count += 1
                            continue

                    response_data = None
                else:
                    # the Text Completion API only requires a prompt and produces 1+ choices/[]/text responses
                    prompt_text = self.get_llm_prompt_text(
                        query=query,
                        suggestion_type=suggestion_type,
                        max_length=self.max_llm_concept_tokens,
                    )

                    token_count = len(self.tokenizer.encode(prompt_text))

                    response = await openai.Completion.acreate(
                        model=model,
                        prompt=prompt_text,
                        max_tokens=4000 - token_count,
                        temperature=0.0,
                    )
                    # set response if present
                    if len(response["choices"]) > 0:
                        try:
                            response_data = json.loads(response["choices"][0]["text"])
                            return response_data
                        except Exception as error:
                            self.logger.error(
                                "Error parsing response from OpenAI Completion API: %s \ndata=%s",
                                error,
                                response,
                            )
                            retry_count += 1
                            continue
                    else:
                        response_data = None
            except Exception as error:
                # log
                self.logger.error(
                    "Error getting response from OpenAI API: %s",
                    error,
                )
                retry_count += 1
                continue

        # return
        if not response_data:
            return {}
        return response_data

    async def search_llm(
        self, query: str, suggestion_type: SuggestionType, num_results: int = 10
    ) -> list[dict]:
        """Search the LLM for the given query.

        Args:
            query (str): Query to search for.
            suggestion_type (SuggestionType): Suggestion type to search.
            num_results (int): Number of results to return.

        Returns:
            list[dict]: List of matching concepts.
        """
        # initialize the results
        results = []

        # get the prompt
        response_data = await self.get_llm_response(query, suggestion_type)

        # get concept labels as a list
        concept_labels = [
            self.lmss.concepts[iri]["label"]
            for iri in self.lmss.get_children(
                self.lmss.key_concepts[suggestion_type.value]
            )
            if self.lmss.concepts[iri]["label"] not in [None, ""]
        ]

        # iterate over the suggestions
        for suggestion, llm_score in response_data:
            # check if the suggestion is in the concepts
            if suggestion not in concept_labels:
                continue

            # check if the suggestion is in the concepts
            for concept in self.lmss.label_to_iri.get(suggestion, []):
                # get the concept
                concept_data = self.lmss.concepts[concept]

                # get exact value
                exact_value = (
                    query.lower() == concept_data["label"].lower()
                    or any(
                        query.lower() == label.lower()
                        for label in concept_data["pref_labels"]
                    )
                    or any(
                        query.lower() == label.lower()
                        for label in concept_data["alt_labels"]
                    )
                    or any(
                        query.lower() == label.lower()
                        for label in concept_data["hidden_labels"]
                    )
                )

                # get startswith value
                startswith_value = (
                    concept_data["label"].lower().startswith(query.lower())
                    or any(
                        label.lower().startswith(query.lower())
                        for label in concept_data["pref_labels"]
                    )
                    or any(
                        label.lower().startswith(query.lower())
                        for label in concept_data["alt_labels"]
                    )
                    or any(
                        label.lower().startswith(query.lower())
                        for label in concept_data["hidden_labels"]
                    )
                )

                # get substring value
                substring_value = (
                    query.lower() in concept_data["label"].lower()
                    or any(
                        query.lower() in label.lower()
                        for label in concept_data["pref_labels"]
                    )
                    or any(
                        query.lower() in label.lower()
                        for label in concept_data["alt_labels"]
                    )
                    or any(
                        query.lower() in label.lower()
                        for label in concept_data["hidden_labels"]
                    )
                )

                results.append(
                    {
                        "iri": concept_data["iri"],
                        "label": concept_data["label"],
                        "match": concept_data["label"],
                        "pref_labels": concept_data["pref_labels"],
                        "alt_labels": concept_data["alt_labels"],
                        "hidden_labels": concept_data["hidden_labels"],
                        "definitions": concept_data["definitions"],
                        "parents": concept_data["parents"],
                        "children": concept_data["children"],
                        "exact": exact_value,
                        "startswith": startswith_value,
                        "substring": substring_value,
                        "score": llm_score,
                        "distance": 1.0 - llm_score,
                    }
                )

        # sort by score descending
        return sorted(
            results,
            key=lambda x: (
                -x["exact"],
                -x["startswith"],
                -x["substring"],
                x["distance"],
            ),
        )[:num_results]

    @staticmethod
    def distance_ensemble(string1: str, string2: str) -> float:
        """Calculate an ensemble of distance metrics with rapidfuzz since we aren't vendoring
        Kelvin NLP and need something decent to use."""
        dist0 = 0.0 if string1.lower() in string2 else 1.0
        dist1 = rapidfuzz.distance.DamerauLevenshtein.normalized_distance(
            string1, string2
        )
        dist2 = rapidfuzz.distance.OSA.normalized_distance(string1, string2)
        dist3 = 1.0 - rapidfuzz.fuzz.token_set_ratio(string1, string2) / 100.0
        dist4 = 1.0 - rapidfuzz.fuzz.partial_token_sort_ratio(string1, string2) / 100.0
        return (dist0 + dist1 + dist2 + dist3 + dist4) / 5.0

    async def suggest_concepts(
        self,
        query: str,
    ) -> list[dict]:
        """Suggest a list of concepts to use in the suggestion engine using get_llm_meta_response(query=text)

        Args:
            query (str): Query to suggest concepts for.

        Returns:
            list[dict]: List of suggested concepts with the iri, label, and target endpoint.
        """

        # match the labels to the key concepts
        results = []

        # iterate through labels for direct match inclusion
        for suggestion_type in list(SuggestionType):
            # search labels
            label_results = self.lmss.search_labels(query, suggestion_type.value, 1)
            if (
                len(label_results) > 0
                and label_results[0]["distance"] < self.max_distance
            ):
                for label_result in label_results:
                    results.append(
                        {
                            "iri": label_result["iri"],
                            "label": label_result["label"],
                            "score": 1.0,
                            "url": CONCEPT_TO_ENDPOINT.get(label_result["label"], None),
                        }
                    )

        # get the list of concepts
        concepts = await self.get_llm_meta_response(query)

        # iterate over the concepts
        for concept, score in concepts:
            if concept in self.lmss.key_concepts:
                concept_iri = self.lmss.key_concepts[concept]
                concept_data = self.lmss.concepts[concept_iri]

                results.append(
                    {
                        "iri": concept_data["iri"],
                        "label": concept_data["label"],
                        "score": score,
                        "url": CONCEPT_TO_ENDPOINT.get(concept, None),
                    }
                )

        # sort by score descending
        return sorted(results, key=lambda x: -x["score"])

    # pylint: disable=R0912,R0915
    async def suggest(
        self,
        query: str,
        suggestion_type: SuggestionType,
        concept_depth: int = 8,
        max_results: int = 5,
        include_parents: bool = True,
        disable_llm: bool = False,
    ) -> list[dict]:
        """Suggest concepts for the given query.

        Args:
            query (str): Query to suggest concepts for.
            suggestion_type (SuggestionType): Suggestion type to use.
            concept_depth (int): Concept depth to use to control granularity of hierarchy
            max_results (int): Maximum number of results to return.
            include_parents (bool): Whether to include parent concepts.
            disable_llm (bool): Whether to disable the LLM.

        Returns:
            list[dict]: List of suggested concepts.
        """
        # initialize the results
        results = []
        seen_iri: set[str] = set()

        # search the labels if enabled
        if self.enable_label_search:
            for result in self.lmss.search_labels(
                search_term=query,
                concept_type=self.lmss.key_concepts[suggestion_type.value],
                concept_depth=concept_depth,
                num_results=max_results,
                include_hidden_labels=self.include_hidden_labels,
                include_alt_labels=self.include_alt_labels,
            ):
                # add it to the results if not seen
                seen_iri.add(result["iri"])
                result["source"] = "label"
                results.append(result)

                if include_parents:
                    for parent in result["parents"]:
                        if parent in seen_iri:
                            continue
                        seen_iri.add(parent)
                        parent_concept = self.lmss.concepts[parent]

                        parent_result = {
                            "iri": parent_concept["iri"],
                            "label": parent_concept["label"],
                            "pref_labels": parent_concept["pref_labels"],
                            "alt_labels": parent_concept["alt_labels"],
                            "hidden_labels": parent_concept["hidden_labels"],
                            "definitions": parent_concept["definitions"],
                            "parents": parent_concept["parents"],
                            "children": parent_concept["children"],
                            "source": "label_parent",
                            "exact": result["exact"],
                            "substring": result["substring"],
                            "startswith": result["starts_with"],
                            "distance": result["distance"],
                            "score": 1.0 - result["distance"],
                        }
                        results.append(parent_result)

        # search the labels if enabled
        if self.enable_definition_search:
            for result in self.lmss.search_definitions(
                search_term=query,
                concept_type=self.lmss.key_concepts[suggestion_type.value],
                concept_depth=concept_depth,
                num_results=max_results,
            ):
                # add it to the results if not seen
                seen_iri.add(result["iri"])
                result["source"] = "definition"
                results.append(result)

                if include_parents:
                    for parent in result["parents"]:
                        if parent in seen_iri:
                            continue
                        seen_iri.add(parent)
                        parent_concept = self.lmss.concepts[parent]

                        parent_result = {
                            "iri": parent_concept["iri"],
                            "label": parent_concept["label"],
                            "pref_labels": parent_concept["pref_labels"],
                            "alt_labels": parent_concept["alt_labels"],
                            "hidden_labels": parent_concept["hidden_labels"],
                            "definitions": parent_concept["definitions"],
                            "parents": parent_concept["parents"],
                            "children": parent_concept["children"],
                            "source": "definition_parent",
                            "exact": result["exact"],
                            "substring": result["substring"],
                            "startswith": result["starts_with"],
                            "distance": result["distance"],
                            "score": 1.0 - result["distance"],
                        }
                        results.append(parent_result)

        # standardize fields
        for result in results:
            if result["label"] is None:
                continue
            result["distance"] = self.distance_ensemble(
                query.lower(), result["label"].lower()
            )
            result["score"] = 1.0 - result["distance"]
            # exact can be for label, alt label, or hidden label
            result["exact"] = (
                query.lower() == result["label"].lower()
                or any(
                    query.lower() == pref_label.lower()
                    for pref_label in result["pref_labels"]
                )
                or any(
                    query.lower() == alt_label.lower()
                    for alt_label in result["alt_labels"]
                )
                or any(
                    query.lower() == hidden_label.lower()
                    for hidden_label in result["hidden_labels"]
                )
            )
            result["substring"] = (
                (query.lower() in result["label"].lower())
                or any(
                    query.lower() in pref_label.lower()
                    for pref_label in result["pref_labels"]
                )
                or any(
                    query.lower() in alt_label.lower()
                    for alt_label in result["alt_labels"]
                )
                or any(
                    query.lower() in hidden_label.lower()
                    for hidden_label in result["hidden_labels"]
                )
                or any(
                    query.lower() in definition.lower()
                    for definition in result["definitions"]
                )
            )
            result["startswith"] = result["label"].lower().startswith(query.lower())
            result["match"] = result["label"]

        # get unique results and sort by score key descending
        results = sorted(
            [result for result in results if result["label"] is not None],
            key=lambda x: (
                -x["exact"],
                -x["startswith"],
                -x["substring"],
                x["distance"],
            ),
        )[:max_results]

        # get the minimum distancem
        min_distance = results[0]["distance"]

        # search the LLM if enabled
        if LLM_ENABLED and self.enable_llm_search and not disable_llm:
            # check for cases to skip
            if (
                (self.min_llm_char_length <= len(query) <= self.max_llm_char_length)
                and (min_distance > self.llm_distance_threshold)
                and not results[0]["startswith"]
                and not results[0]["substring"]
            ):
                if self.llm_distance_threshold <= min_distance:
                    # search the llm if the result quality is too low
                    if len(results) == 0 or min_distance > self.max_distance:
                        results += await self.search_llm(
                            query=query,
                            suggestion_type=suggestion_type,
                            num_results=max_results,
                        )
                        for result in results:
                            result["source"] = "llm"

        # only keep the best result for each iri
        iri_min: dict[str, dict] = {}
        for result in results:
            # check that this is the best result for the iri
            if result["iri"] in iri_min:
                if result["distance"] > iri_min[result["iri"]]["distance"]:
                    continue
            iri_min[result["iri"]] = result

        results = list(iri_min.values())

        # standardize fields
        for result in results:
            if result["source"] != "llm":
                result["distance"] = self.distance_ensemble(
                    query.lower(), result["label"].lower()
                )
                result["score"] = 1.0 - result["distance"]

        # get unique results and sort by score key descending
        results = sorted(
            results,
            key=lambda x: (
                -x["exact"],
                -x["startswith"],
                -x["substring"],
                x["distance"],
            ),
        )[:max_results]

        return results


telly.go(sys.modules[__name__], '26d17932-7a3d-4e12-9beb-f48993941f30')
