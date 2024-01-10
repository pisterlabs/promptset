
import logging
import sys

from hubmap_sdk import SearchSdk
import json
from collections import defaultdict
import random
import pathlib

from pathlib import Path

from git import Repo
from langchain.document_loaders import GitLoader, DirectoryLoader
from langchain.docstore.document import Document
import requests
from langchain.embeddings import OpenAIEmbeddings, FakeEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.callbacks import get_openai_callback
from joblib import Parallel, delayed
import json
import tiktoken

from enum import Enum


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

SEARCH_API_URL = "https://search.api.hubmapconsortium.org/v3/"

SEARCHSDK_INSTANCE = SearchSdk(service_url=SEARCH_API_URL)

PERSIST_DIR = Path("persist")

def get_chroma_persist_dir(entity_type):
    return PERSIST_DIR / "chroma_es_field_db" / entity_type.lower()
    

DEBUG_OUT_DIR = Path("debug_out")

CHROMA_COL_NAME = "default"

EMBEDDINGS = OpenAIEmbeddings()
ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")


MAX_TOKENS_PER_DOC = 400

INCLUDELIST_DATASET = [
    "created_timestamp",
    "donor.hubmap_id",
    "donor.mapped_metadata.age_value",
    "donor.mapped_metadata.blood_type",
    "donor.mapped_metadata.body_mass_index_value",
    "donor.mapped_metadata.cause_of_death",
    "donor.mapped_metadata.death_event",
    "donor.mapped_metadata.height_value",
    "donor.mapped_metadata.kidney_donor_profile_index_value",
    "donor.mapped_metadata.kidney_donor_profile_index_value[n]",
    "donor.mapped_metadata.mechanism_of_injury",
    "donor.mapped_metadata.medical_history",
    "donor.mapped_metadata.race",
    "donor.mapped_metadata.sex",
    "donor.mapped_metadata.social_history",
    "donor.mapped_metadata.weight_value",
    "donor.uuid",
    "files[n].rel_path",
    "files[n].size",
    "files[n].type",
    "group_name",
    "group_uuid",
    "hubmap_id",
    "mapped_data_types",
    "origin_samples[n].mapped_organ",
    "published_timestamp",
    "source_samples[n].created_timestamp",
    "source_samples[n].group_uuid",
    "source_samples[n].mapped_sample_category",
    "source_samples[n].metadata.health_status",
    "source_samples[n].metadata.perfusion_solution",
    "source_samples[n].metadata.specimen_preservation_temperature",
    "source_samples[n].metadata.vital_state",
    "source_samples[n].tissue_type",
    "uuid",
]

EXCLUDELIST_DATASET = [ 
    "ancestors",
    "descendants",
    "immediate_ancestors",
    "immediate_descendants",
    "metadata",
    "donor.metadata",
    "title",
    "*.filepath",
    "*.rui_location",
    "*description",
    "*url",
    "dataset_info",
    "*displayname",
    "contributors",
    "contacts",
    # "*rel_path",
    "*provider_info",
    "*uuid",
    "*_id",
    "*_ids",
    "*counts*",
    "mapper_metadata",
]

WILDCARDLIST = [
    "files[n].rel_path",
]


class EntityType(str, Enum):
    DATASET = "Dataset"
    DONOR = "Donor"
    SAMPLE = "Sample"


ENTITY_TYPES = [
    EntityType.DATASET,
    EntityType.DONOR,
    EntityType.SAMPLE,
]




EXCLUDELIST_MAP = {
    EntityType.DATASET: EXCLUDELIST_DATASET,
    EntityType.DONOR: [],
    EntityType.SAMPLE: [],
}

INCLUDELIST_MAP = {
    EntityType.DATASET: INCLUDELIST_DATASET,
    EntityType.DONOR: [],
    EntityType.SAMPLE: [],
}


SEARCH_MAX_RESULTS = 300

def get_search_template(entity_type):
    return {
        "query": {
            "function_score": {
                "query": {
                    "match": {
                        "entity_type": entity_type
                    }
                },
                "random_score": {
                    "seed": 10
                }
            },
        },
    "_source": {
        "excludes": EXCLUDELIST_DATASET,
    },
    "size": SEARCH_MAX_RESULTS,
    }

DOC_FORMAT = """{key} example values: {extras}
{value}"""


def num_tokens_from_string(string: str) -> int:
    num_tokens = len(ENCODING.encode(string))
    return num_tokens


def aggregate_field_examples(search_result):
    field_example_dict = defaultdict(lambda: defaultdict(int))

    def get_field_example_dict(json_obj, field_example_dict, prefix=""):
        if isinstance(json_obj, dict):
            for key in json_obj:
                if prefix == "":
                    new_prefix = key
                else:
                    new_prefix = prefix + "." + key
                get_field_example_dict(json_obj[key], field_example_dict, new_prefix)
        elif isinstance(json_obj, list):
            for i, item in enumerate(json_obj):
                new_prefix = prefix + "[n]"
                get_field_example_dict(item, field_example_dict, new_prefix)
        else:
            if "rel_path" in prefix:
                json_obj = pathlib.Path(json_obj).suffix
                if len(json_obj) > 0:
                    json_obj = "*"+json_obj
            field_example_dict[prefix][json_obj] += 1


    for hit in search_result["hits"]["hits"]:
        get_field_example_dict(hit["_source"], field_example_dict)

    return field_example_dict



def aggregated_field_examples_to_docs(field_example_dict, entity_type):
    docs = []
    for k, v in field_example_dict.items():
        # if a \n appears in the any string, skip it
        if any([True for x in v if '\n' in str(x)]):
            continue
        
        dual_list = [(a, b) for a, b in v.items()]
        # filter out empty
        dual_list = [(a, b) for a, b in dual_list if a != ""]

        if len(dual_list) == 0:
            continue

        total_hits = sum([b for a, b in dual_list])
        max_hits = max([b for a, b in dual_list])
        # print(total_hits)
        if total_hits < SEARCH_MAX_RESULTS / 4:
            print(f"Skipping {k} because it has {total_hits} hits")
            continue
        # dual_list.sort(key=lambda x: x[1], reverse=True)
        random.shuffle(dual_list)
        # dual_list.sort(key=lambda x: x[0])

        max_tokens_this_doc = MAX_TOKENS_PER_DOC
        
        if max_hits == 1:
            max_tokens_this_doc = int(MAX_TOKENS_PER_DOC/4)

        docval = '\n'.join([f"{a}" for a, b in dual_list]) + '\n'
        docval_encoded = ENCODING.encode(docval)
        docval_cropped = ENCODING.decode(docval_encoded[:max_tokens_this_doc])
        docval_cropped = docval_cropped[:docval_cropped.rfind('\n')]

        # if not beginswith INCLUDELIST, skip
        if not any([k.startswith(x) for x in INCLUDELIST_MAP[entity_type]]):
            continue

        extras = ' (not a nested type) (never use in "filter", use "must")'
        if "size" in k:
            extras += " (file size in bytes)"

        if any([isinstance(a, str) and "*" in a for a, b in dual_list]):
            extras += ' (use "bool" "must" "wildcard")'
        elif all([(isinstance(a, str) and a.replace('.', '', 1).isdigit()) or isinstance (a, int) or isinstance(a, float) for a, b in dual_list]):
            extras += ' (use "bool" "must" "range")'
        else:
            extras += ' (use "bool" "must" "match")'
        
        docs.append(Document(page_content=DOC_FORMAT.format(key=k, value=docval_cropped, extras=extras)))



def print_docs_stats(docs):
    tokens_per_doc = [num_tokens_from_string(doc.page_content) for doc in docs]
    num_tokens = sum(tokens_per_doc)
    print(f"Total number of tokens: {num_tokens}")
    embedding_price_per_token = 0.0001 / 1000
    print(f"Total cost: ${num_tokens * embedding_price_per_token:.3f}")
    tokens_per_doc = [num_tokens_from_string(doc.page_content) for doc in docs]
    print(f"Average tokens per doc: {sum(tokens_per_doc) / len(tokens_per_doc)}")
    print(f"Max tokens in one doc: {max(tokens_per_doc)}")
    print(f"Index of max tokens: {tokens_per_doc.index(max(tokens_per_doc))}")
    print(f"Content of doc with max tokens:\n{docs[tokens_per_doc.index(max(tokens_per_doc))].page_content}")



def overwrite_chroma_db(persist_directory, collection_name, docs):
    print(f"loading database from {persist_directory} with collection name {collection_name}")
    vectordb = Chroma(persist_directory=str(persist_directory), embedding_function=EMBEDDINGS, collection_name=collection_name)
    vectordb._client.reset()

    vectordb = Chroma.from_documents(documents=docs, embedding=EMBEDDINGS, persist_directory=str(persist_directory), collection_name=collection_name)

    vectordb.persist()
    vectordb = None

def generate_es_field_db(entity_type):
    search_result = SEARCHSDK_INSTANCE.search_by_index(get_search_template(entity_type), "portal")
    field_example_dict = aggregate_field_examples(search_result)
    docs = aggregated_field_examples_to_docs(field_example_dict, entity_type)

    with open(DEBUG_OUT_DIR/f"es_field_db_{entity_type}.txt", "w") as f:
        for d in docs:
            f.write(d.page_content)
            f.write("\n\n")

    with open(DEBUG_OUT_DIR/f"es_field_db_{entity_type}.json", "w") as f:
        json.dump(field_example_dict, f, indent=2)

    overwrite_chroma_db(get_chroma_persist_dir(entity_type), CHROMA_COL_NAME, docs)

if __name__ == "__main__":
    ENTITY_TYPE = ENTITY_TYPES[0]

    generate_es_field_db(ENTITY_TYPE)






