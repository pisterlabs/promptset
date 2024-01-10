import logging
from typing import List

import openai
from tqdm import tqdm

from ds.sentence_item import SentenceItem
from pipeline.pipeline_component import PipelineComponent
from utils.mongodb_handler import get_database

logger = logging.getLogger(__name__)


def build_prompt(cluster, domain=None, place=None):
    distinct_sents = sorted(list(set(s.text for s in cluster)),
                            key=lambda x: len(x))
    prompt = []
    if domain and place:
        prompt.append(
            f"Given the following sentences about {domain} in {place}:\n")
    else:
        prompt.append(f"Given the following sentences sorted by length:\n")
    for i, st in enumerate(distinct_sents):
        prompt.append(f'({i + 1}) {st}')
        # to avoid too long prompts exceeding OpenAI's limit (2048)
        if len("\n".join(prompt)) > 1800:
            break
    prompt.append("\nSummarize them using one sentence:")
    prompt = "\n".join(prompt)
    return prompt


class RepresentativeGenerator(PipelineComponent):
    description = "Cluster statements"
    config_layer = ["pipeline_components", "representative_generator"]

    def __init__(self, config):
        super().__init__(config)

        # Get local config
        self._local_config = config
        for layer in self.config_layer:
            self._local_config = self._local_config[layer]

        self._node_ids = self._local_config["input"]["ids"]
        self._target_label = self._local_config["input"]["label"]

        openai.api_key_path = self._local_config["openai_api_key_path"]

        # Get the database config
        db_config = self._local_config["db_collections"]

        # Assign the database collections
        db = get_database(**config["mongo_db"])
        self._sentences_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['sentences']['name']}"]
        self._clusters_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['clusters']['name']}_{self._target_label}"]
        self._clusters_with_reps_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['clusters_with_reps']['name']}_{self._target_label}"]

        # Index the collection
        for index in db_config["clusters_with_reps"]["indexes"]:
            field = index["field"]
            unique = index.get("unique", False)
            self._clusters_with_reps_collection.create_index(field,
                                                             unique=unique)

    def run(self):
        logger.info("Running representative generator")

        logger.info("Querying clusters")
        cluster_items = list(self._clusters_collection.find(
            {"node_id": {"$in": self._node_ids}}
        ))

        existing_node_ids = set(
            self._clusters_with_reps_collection.distinct("node_id"))

        logger.info(
            f"Generating representatives for {len(cluster_items)} nodes")

        overwrite = self._local_config['overwrite']
        for c in cluster_items:
            if c["node_id"] in existing_node_ids:
                logger.info(
                    f"Node {c['node_id']} is already in the database, "
                    f"overwriting is set to {overwrite}")
                if overwrite:
                    logger.info(
                        f"Deleting all existing data for node {c['node_id']}")
                    self._clusters_with_reps_collection.delete_many(
                        {"node_id": c["node_id"]})
                else:
                    logger.info(
                        f"Skipping node {c['node_id']}")
                    continue
            self.generate_cluster_representatives(c)

    def needs_spacy_docs(self) -> bool:
        return False

    def initialize(self):
        return

    def needs_people_group_tree(self) -> bool:
        return False

    def is_initialized(self) -> bool:
        return True

    def generate_cluster_representatives(self, cluster_item):
        logger.info(
            f"Generating representatives for node {cluster_item['node_id']}")
        raw_clusters = [c for c in cluster_item["clusters"][
                                   :self._local_config["cluster_limit"]] if
                        len(c) >= self._local_config["min_cluster_size"]]

        clusters = []
        for c in raw_clusters:
            clusters.append([SentenceItem.from_dict(s) for s in
                             self._sentences_collection.find(
                                 {"_id": {"$in": c}})])

        reps = self.generate_cluster_representative(clusters)

        results = []
        for cluster, rep in zip(clusters, reps):
            # rep = get_first_sentence(rep)
            # if not rep.endswith("."):
            #     continue
            # if len(rep) > 100:
            #     continue
            # if len(rep) < 20:
            #     continue
            # if any(rep.startswith(p) for p in
            #        ["The sentences ", "The sentence "]):
            #     continue
            # pattern = re.compile(r"(\(\d+\)|(sentence \d+))", re.IGNORECASE)
            # if pattern.search(rep):
            #     continue
            #
            # distinct_sents = sorted(set(s.text.lower() for s in cluster),
            #                         key=lambda x: len(x), reverse=True)
            # count = len(distinct_sents)
            #
            # if count == 1:
            #     continue
            #
            # if count / len(cluster) < 1 / 3:
            #     continue
            #
            # tokens = rep.split()
            # if len(tokens) < 3:
            #     continue

            results.append({
                "node_id": cluster_item["node_id"],
                "sentence_item_ids": [c.get_id() for c in cluster],
                "rep": rep
            })

        # cluster_with_reps = {
        #     "node_id": cluster_item["node_id"],
        #     "clusters": results,
        # }

        # Insert one-by-one just in case there's disruption from OpenAI APIs
        if len(results) > 0:
            logger.info(
                f"Insert to DB: node id {cluster_item['node_id']}, "
                f"number of clusters {len(results)}")
            self._clusters_with_reps_collection.insert_many(results)
        else:
            logger.info("No representatives to insert")

        # return cluster_with_reps

    def generate_cluster_representative(self,
                                        clusters: List[List[SentenceItem]]) \
            -> List[str]:
        reps = []
        for cluster in tqdm(clusters):
            distinct_sents = sorted(set(s.text for s in cluster),
                                    key=lambda x: len(x), reverse=False)
            if len(set(s.lower() for s in distinct_sents)) <= 2:
                rep = distinct_sents[0]
            else:
                prompt = build_prompt(cluster)
                response = openai.Completion.create(
                    engine=self._local_config["gpt3_model"],
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=50,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    # stop=["."]
                )
                rep = response["choices"][0]["text"].strip()  # + "."
            reps.append(rep)
        return reps
