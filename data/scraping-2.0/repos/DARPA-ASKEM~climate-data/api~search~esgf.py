from api.settings import default_settings
from api.search.provider import BaseSearchProvider, DatasetSearchResults, Dataset
import requests
from urllib.parse import urlencode
from typing import List, Dict, Any
import itertools
import dask
from openai import OpenAI
from numpy import dot
from numpy.linalg import norm
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

NATURAL_LANGUAGE_PROCESSING_CONTEXT = """
You are a tool to extract search terms by category from a given search request. 

The given fields are: frequency, nominal_resolution, lower_time_bound, upper_time_bound, and description.

The definitions of the fields are as follows. 

frequency is a duration. 
Possible example values for frequency values are: 6 hours, 6hrs, 3hr, daily, day, yearly, 12 hours, 12 hr

nominal_resolution is a measure of distance. 
Possible example values for resolution are: 100 km, 100km, 200km, 200 km, 2x2 degrees, 1x1 degrees, 1x1, 2x2, 20000 km

lower_time_bound and upper_time_bound are measures of time. 
When extracted from user input, convert them to UTC ISO 8601 format.
Possible example values include:

"after 2022" = lower_time_bound: 2022-00-00T00:00:00Z
"between march 2021 and april 2023" = lower_time_bound: 2021-03-00T00:00:00Z ; upper_time_bound: 2023-04-00T00:00:00Z
"before september 1995" = upper_time_bound: 1995-09-00T00:00:00Z

description is a text field and contains all other unprocessed information. 

Return the fields as a JSON object and include no other information. 

Examples of full processing are as follows. 

Input:
100km before 2023 daily air temperature

Output:
{
    "frequency": "daily",
    "nominal_resolution": "100km",
    "upper_time_bound": "2023-00-00T00:00:00Z",
    "description": "air temperature"
}

Input: 2x2 degree relative humidity between june 1997 and july 1999 6hr

Output:
{
    "frequency": "6hr",
    "nominal_resolution": "2x2 degree",
    "lower_time_bound": "1997-06-00T00:00:00Z",
    "upper_time_bound": "1999-07-00T00:00:00Z",
    "description": "relative humidity"
}
"""

# cosine matching threshold to greedily take term
GREEDY_EXTRACTION_THRESHOLD = 0.95

SEARCH_FACETS = {
    # match by cosine similarity
    "similar": [
        "experiment_title",
        "cf_standard_name",
        "variable_long_name",
        "variable_id",
        "table_id",
    ],
    # only take exact matches
    "exact": [
        "institution_id",
        "master_id",
        "variant_label",
        "experiment_id",
    ],
    # create embeddings, but handle manually elsewhere
    "other": ["nominal_resolution", "frequency"],
}


class ESGFProvider(BaseSearchProvider):
    def __init__(self, openai_client):
        print("initializing esgf search provider")
        self.client: OpenAI = openai_client
        self.embeddings = {}

    def initialize_embeddings(self, force_refresh=False):
        """
        creates string embeddings if needed, otherwise reloads from cache.
        force_refresh is needed if the list of facets changes.
        """
        cache = Path("./embedding_cache")
        if cache.exists() and not force_refresh:
            print("embedding cache exists", flush=True)
            with cache.open("rb") as f:
                self.embeddings = pickle.load(f)
        else:
            print("no embedding cache, generating new", flush=True)
            with cache.open(mode="wb") as f:
                self.embeddings = self.extract_embedding_strings()
                pickle.dump(self.embeddings, f)

    def search(
        self, query: str, page: int, force_refresh_cache: bool = False
    ) -> DatasetSearchResults:
        """
        converts a natural language query to a list of ESGF dataset
        metadata dictionaries by running a lucene query against the given
        ESGF node in settings.
        """
        if len(self.embeddings.keys()) == 0:
            self.initialize_embeddings(force_refresh_cache)
        return self.natural_language_search(query, page)

    def get_access_urls_by_id(self, dataset_id: str) -> List[str]
        """
        returns a list of OPENDAP URLs for use in processing given a dataset.
        """
        params = urlencode(
            {
                "type": "File",
                "format": "application/solr+json",
                "dataset_id": dataset_id,
                "limit": 50,
            }
        )
        full_url = f"{default_settings.esgf_url}/search?{params}"
        r = requests.get(full_url)
        response = r.json()
        if r.status_code != 200:
            raise ConnectionError(
                f"Failed to extract files from dataset via file search: {full_url} {response}"
            )
        files = response["response"]["docs"]
        if len(files) == 0:
            raise ConnectionError(
                f"Failed to extract files from dataset: empty list {full_url}"
            )
        # file url responses are lists of strings with their protocols separated by |
        # e.x. https://esgf-node.example|mimetype|OPENDAP
        opendap_urls = [
            url.split("|")[0]
            for url in itertools.chain.from_iterable([f["url"] for f in files])
            if "OPENDAP" in url
        ]
        # sometimes the opendap request form is returned. we strip the trailing suffix if needed
        opendap_urls = [u[:-5] if u.endswith(".nc.html") else u for u in opendap_urls]
        return opendap_urls

    def get_access_urls(self, dataset: Dataset) -> List[str]:
        return self.get_access_urls_by_id(dataset.metadata["id"])

    def natural_language_search(
        self, search_query: str, page: int
    ) -> DatasetSearchResults:
        """
        converts to natural language and runs the result against the ESGF node, returning a list of datasets.
        """
        search_terms_json = self.process_natural_language(search_query)
        search_terms = json.loads(search_terms_json)

        query = self.generate_query_string(search_terms)
        options = self.generate_temporal_coverage_query(search_terms)

        print(query, flush=True)

        return self.run_esgf_query(query, page, options)

    def build_natural_language_prompt(self, search_query: str) -> str:
        """
        wraps user input given to the LLM after the context.
        """
        return "Convert the following input text: {}".format(search_query)

    def process_natural_language(self, search_query: str) -> str:
        """
        runs query against LLM and returns the result string.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": NATURAL_LANGUAGE_PROCESSING_CONTEXT},
                {
                    "role": "user",
                    "content": self.build_natural_language_prompt(search_query),
                },
            ],
            temperature=0.7,
        )
        query = response.choices[0].message.content
        return query

    def run_esgf_query(
        self, query_string: str, page: int, options: Dict[str, str]
    ) -> DatasetSearchResults:
        """
        runs the formatted apache lucene query against the ESGF node and returns the metadata in datasets.
        """
        encoded_string = urlencode(
            {
                "query": query_string,
                "project": "CMIP6",
                "fields": "*",
                "latest": "true",
                "sort": "true",
                "limit": f"{default_settings.entries_per_page}",
                "offset": "{}".format(default_settings.entries_per_page * (page - 1)),
                "format": "application/solr+json",
                "distrib": "true",
            }
            | options
        )

        full_url = f"{default_settings.esgf_url}/search?{encoded_string}"
        r = requests.get(full_url)
        response = r.json()
        if r.status_code != 200:
            raise ConnectionError(
                f"Failed to search against ESGF node: {full_url} {r.status_code} {response}"
            )

        # parallel over datasets, but delay fetching url until needed
        return dask.compute(
            [
                dask.delayed(Dataset)(
                    metadata,
                )
                for metadata in response["response"]["docs"]
            ]
        )[0]

    def get_embedding(self, text):
        """returns an embedding for a single string."""
        return (
            self.client.embeddings.create(input=[text], model="text-embedding-ada-002")
            .data[0]
            .embedding
        )

    def get_embeddings(self, text):
        """returns a list of embeddings for a list of strings."""
        return [
            e.embedding
            for e in self.client.embeddings.create(
                input=text, model="text-embedding-ada-002"
            ).data
        ]

    def cosine_similarity(self, a, b):
        return dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def extract_embedding_strings(self) -> Dict[str, pd.DataFrame]:
        """
        builds embeddings dictionary given the desired SEARCH_FACETS. finds possible
        values for given facets from ESGF node and then gets string embeddings of those
        enumerated values.
        """
        desired_facets = [item for inner in SEARCH_FACETS.values() for item in inner]
        print(desired_facets)
        encoded_string = urlencode(
            {
                "project": "CMIP6",
                "facets": ",".join(desired_facets),
                "limit": "0",
                "format": "application/solr+json",
            }
        )
        facet_possibilities = (
            f"https://esgf-node.llnl.gov/esg-search/search?{encoded_string}"
        )

        print("querying fields", flush=True)
        r = requests.get(facet_possibilities)
        response = r.json()
        if r.status_code != 200:
            raise ConnectionError(
                f"Failed to get facet potential values from ESGF node: {facet_possibilities} {response}"
            )
        fields = response["facet_counts"]["facet_fields"]
        print("aggregating fields", flush=True)
        embeddings = {
            f: pd.DataFrame({"string": fields[f][::2]}) for f in desired_facets
        }
        print("creating embeddings...", flush=True)
        for k in embeddings.keys():
            print(f"  embeddings for: {k}", flush=True)
            # drop '' and other falsy strings
            embeddings[k] = embeddings[k][embeddings[k].string.astype(bool)]
            embeddings[k]["embed"] = self.get_embeddings(embeddings[k].string.to_list())

        return embeddings

    def extract_relevant_description(self, description: str) -> List[str]:
        """
        takes the LLM-extracted description field and parses it into meaningful
        terms to build into the formatted apache lucene query.
        """
        # experiment id and variant id are best taken as exact match rather than assumed by cosine
        # general idea:
        #   break on word boundary and take...
        #     exact matches to experiment id and variant_label
        #     anything that's over GREEDY_EXTRACTION_THRESHOLD
        #   otherwise...
        #     take non-matching inputs and conjoin them back into a phrase to take highest match across all categories
        #     take the most relevant between averaged individual token similarities and the whole phrase
        tokens = description.replace(",", " ").split()
        matched = []
        exact_match_values = [
            match
            for nested_list in [
                self.embeddings[field].string.values for field in SEARCH_FACETS["exact"]
            ]
            for match in nested_list
        ]

        fallback_similarities = []

        print(f"finding best terms for {tokens}")

        def get_single_best_match(text, similar_fields):
            most_similar = ("", 0.00)
            for field in similar_fields:
                embedding = self.get_embedding(text)
                self.embeddings[field]["similarities"] = self.embeddings[field][
                    "embed"
                ].apply(lambda e: self.cosine_similarity(e, embedding))
                best_match = (
                    self.embeddings[field]
                    .sort_values("similarities", ascending=False)
                    .head(3)
                )
                string = best_match.string.values[0]
                similarity = best_match.similarities.values[0]
                print(f"    {string} => {similarity}")
                if similarity >= most_similar[1]:
                    most_similar = (string, similarity)
            return most_similar

        # clone tokens to remove during iteration

        for t in tokens[:]:
            if t in exact_match_values:
                print(f"  exact match: {t}")
                matched.append(t)
                tokens.remove(t)
            else:
                print(f"  approximate matching for {t}")
                phrase, similarity = get_single_best_match(t, SEARCH_FACETS["similar"])
                if similarity >= GREEDY_EXTRACTION_THRESHOLD:
                    print(
                        f"    matched word {t} -> {phrase} over threshold {GREEDY_EXTRACTION_THRESHOLD}: {similarity}"
                    )
                    matched.append(phrase)
                    tokens.remove(t)
                else:
                    print(
                        f"    closest match {t} -> {phrase} is under threshold {GREEDY_EXTRACTION_THRESHOLD}: {similarity}"
                    )
                fallback_similarities.append((phrase, similarity))

        if len(tokens) == 0:
            return matched

        print(f"  leftover tokens: {tokens}\nmatching for whole phrase")

        conjoined_phrase, conjoined_similarity = get_single_best_match(
            " ".join(tokens), SEARCH_FACETS["similar"]
        )

        fallback_similarities = [f for f in fallback_similarities if f[0] in tokens]
        avg_sim = sum((map(lambda f: f[1], fallback_similarities))) / len(
            fallback_similarities
        )
        print(f"  conjoined similarity {conjoined_similarity} - avg by parts {avg_sim}")
        if conjoined_similarity >= avg_sim:
            print(f"    using conjoined phrase {conjoined_phrase}")
            matched.append(conjoined_phrase)
        else:
            for part in fallback_similarities:
                matched += part[0]

        print(f"finalized search terms are {matched}")

        return matched

    def generate_query_string(self, search_terms: Dict[str, str]) -> str:
        """
        handles LLM-extracted fields separately as needed and returns the formatted lucene query.
        """
        desired_terms = ["nominal_resolution", "frequency"]
        best_matches = []
        for desired in desired_terms:
            if desired in search_terms:
                embedding = self.get_embedding(search_terms[desired])
                self.embeddings[desired]["similarities"] = self.embeddings[desired][
                    "embed"
                ].apply(lambda e: self.cosine_similarity(e, embedding))
                print(f"querying for {desired}: {search_terms[desired]}")
                print(
                    self.embeddings[desired]
                    .sort_values("similarities", ascending=False)
                    .head(3),
                    flush=True,
                )
                best_matches.append(
                    self.embeddings[desired]
                    .sort_values("similarities", ascending=False)
                    .head(1)["string"]
                    .values[0]
                )
        description = []
        if "description" in search_terms:
            description = self.extract_relevant_description(search_terms["description"])
        query_string = " AND ".join(map(lambda t: f'"{t}"', best_matches + description))
        print(f"lucene query: {query_string}")
        return query_string

    def generate_temporal_coverage_query(self, terms: Dict[str, str]) -> Dict[str, str]:
        """
        creates ESGF search time bound arguments.
        """
        query = {}
        if "upper_time_bound" in terms:
            query["end"] = terms["upper_time_bound"]
        if "lower_time_bound" in terms:
            query["start"] = terms["lower_time_bound"]
        return query
