import os
import pickle as pkl
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import cosine_similarity
import cohere
from utils.utils import extract_floats_from_table, find_shared_floats


class TableMatcher:
    def __init__(
            self,
            dir_table_to_embedding: str,
            dir_table_match: str,
            dir_source_papers: str,
            format: str = "html",
            embedding_model: str = "ada-002",
            weighting: bool = False
    ):
        assert os.path.exists(dir_table_match), f"{dir_table_match} does not exist."
        assert os.path.exists(dir_source_papers), f"{dir_source_papers} does not exist."
        assert embedding_model in [
            "ada-002", "embed-english-light-v2.0", "embed-english-v2.0", "embed-multilingual-v2.0"
        ]

        self.dir_table_to_embedding: str = dir_table_to_embedding
        self.dir_table_match: str = dir_table_match
        self.dir_source_papers: str = dir_source_papers
        self.format: str = format
        self.embedding_model: str = embedding_model
        self.threshold: float = self.get_threshold(embedding_model)
        self.weighting: bool = weighting

    @staticmethod
    def get_threshold(embedding_model: str) -> float:
        if embedding_model == "ada-002":
            return 0.30
        elif embedding_model == "embed-english-light-v2.0":
            return 0.25
        elif embedding_model == "embed-english-v2.0":
            return 0.35
        elif embedding_model == "embed-multilingual-v2.0":
            return 0.35
        else:
            raise ValueError

    @staticmethod
    def convert_html_to_markdown(table: str) -> str:
        soup = BeautifulSoup(
            table.replace("</thead>", '').replace("<tbody>", '').replace("<thead>", "<tbody>"), 'html.parser'
        )
        try:
            df = pd.read_html(str(soup))[0]
        except ValueError:
            return table
            # raise ValueError
        return df.to_markdown()

    @staticmethod
    def convert_html_to_csv(table: str) -> str:
        soup = BeautifulSoup(
            table.replace("</thead>", '').replace("<tbody>", '').replace("<thead>", "<tbody>"), 'html.parser'
        )
        df = pd.read_html(str(soup))[0]
        return df.to_csv()

    def load_tables(self, target_arxiv_id: str, source_arxiv_id: str, is_target: bool) -> List[str]:
        if is_target:
            fp_table_match: str = f"{self.dir_table_match}/{target_arxiv_id}/{source_arxiv_id}.json"
            table_match: Dict[str, str] = json.load(open(fp_table_match, "r"))

            target_table = table_match["target_table"]
            if self.format == "markdown":
                target_table = self.convert_html_to_markdown(target_table)
            elif self.format == "csv":
                target_table = self.convert_html_to_csv(target_table)
            target_caption = table_match["target_caption"]

            list_tables: List[str] = [target_table + '\n' + target_caption]
            list_original_tables: List[str] = [table_match["target_table"] + '\n' + table_match["target_caption"]]
        else:
            fp_source_paper_info: str = f"{self.dir_source_papers}/{source_arxiv_id}.json"
            source_paper_info = json.load(open(fp_source_paper_info, 'r'))
            list_table_and_caption: List[dict] = source_paper_info["list_table_and_caption"]
            
            list_tables: List[str] = list()
            list_original_tables: List[str] = list()
            for table_and_caption in list_table_and_caption:
                table = table_and_caption["table"]
                if self.format == "markdown":
                    table = self.convert_html_to_markdown(table)
                elif self.format == "csv":
                    table = self.convert_html_to_csv(table)
                list_tables.append(table + '\n' + table_and_caption["caption"])
                list_original_tables.append(table_and_caption["table"] + '\n' + table_and_caption["caption"])
            
        return list_tables, list_original_tables

    @staticmethod  # https://platform.openai.com/docs/guides/embeddings/use-cases
    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    @staticmethod
    def get_cohere_embedding(text, model="embed-english-light-v2.0") -> List[float]:
        assert model in ["embed-english-light-v2.0", "embed-english-v2.0", "embed-multilingual-v2.0"]

        api_key = os.environ["COHERE_API_KEY"]

        # Create and retrieve a Cohere API key from dashboard.cohere.ai
        co = cohere.Client(api_key)

        # Embed the training set
        embedding: List[float] = co.embed(texts=[text], model=model).embeddings[0]
        return embedding

    def get_table_to_embedding(
            self,
            target_arxiv_id: str,
            source_arxiv_id: str,
            is_target: bool
    ) -> Union[Tuple[Dict[str, np.ndarray], str, str], Dict[str, np.ndarray]]:
        if is_target:
            # note that a target table can vary with a source paper
            fp_table_to_embedding: str = f"{self.dir_table_to_embedding}/target_table/{self.embedding_model}/{self.format}/{target_arxiv_id}/{source_arxiv_id}.pkl"

            fp_table_match: str = f"{self.dir_table_match}/{target_arxiv_id}/{source_arxiv_id}.json"
            table_match: Dict[str, str] = json.load(open(fp_table_match, "r"))

            gt_target_table = table_match["target_table"] + '\n' + table_match["target_caption"]
            gt_source_table = table_match["source_table"] + '\n' + table_match["source_caption"]

        else:
            # note that this is not a table, but a paper and it contains multiple candidate tables and their embeddings
            fp_table_to_embedding: str = f"{self.dir_table_to_embedding}/source_paper/{self.embedding_model}/{self.format}/{source_arxiv_id}.pkl"

        try:
            table_to_embedding: Dict[str, np.ndarray] = pkl.load(open(fp_table_to_embedding, "rb"))
        except FileNotFoundError:
            list_tables, list_original_tables = self.load_tables(
                target_arxiv_id=target_arxiv_id, source_arxiv_id=source_arxiv_id, is_target=is_target
            )
            table_to_embedding: Dict[str, np.ndarray] = dict()
            for table, otable in zip(list_tables, list_original_tables):
                if self.embedding_model == "ada-002":
                    embedding: np.array = np.array(self.get_embedding(text=table))
                elif self.embedding_model == "cohere":
                    embedding: np.array = np.array(self.get_cohere_embedding(text=table))
                else:
                    raise NotImplementedError
                table_to_embedding[otable] = embedding

            os.makedirs(os.path.dirname(fp_table_to_embedding), exist_ok=True)
            pkl.dump(table_to_embedding, open(fp_table_to_embedding, "wb"))

        if is_target:
            return table_to_embedding, gt_target_table, gt_source_table
        else:
            return table_to_embedding

    def match(
            self,
            target_embeddings: np.ndarray,
            source_embeddings: np.ndarray,
            target_table: str,
            list_candidate_source_tables: List[str] = None,
    ):
        cos_sim = cosine_similarity(target_embeddings, source_embeddings.T).squeeze(axis=0)  # (1, N) -> (N, )
        assert -1 <= cos_sim.min() <= cos_sim.max() <= 1, \
            f"cos_sim.min(): {cos_sim.min()}, cos_sim.max(): {cos_sim.max()}"

        list_target_floats = extract_floats_from_table(target_table)
        if self.weighting:
            list_n_shared_floats = []
            for table in list_candidate_source_tables:
                list_source_floats = extract_floats_from_table(table)
                list_shared_floats = find_shared_floats(list1=list_target_floats, list2=list_source_floats)
                n_shared_floats = len(list_shared_floats)
                list_n_shared_floats.append(n_shared_floats)
            array_n_shared_floats = np.array(list_n_shared_floats)

            sorted_indices = np.argsort(-array_n_shared_floats)  # sort in descending order

            # count the number of non-zero elements in array_n_shared_floats
            n_non_zeros = np.sum(array_n_shared_floats != 0)

            weights: dict = defaultdict(float)
            linear_weights = np.linspace(1.0, 1.0 / n_non_zeros, n_non_zeros)

            for i, sorted_index in enumerate(sorted_indices):
                if list_n_shared_floats[sorted_index] > 0:
                    weights[sorted_index] = linear_weights[i]
                else:
                    weights[sorted_index] = 0.0
            weights: np.ndarray = np.array([v for k, v in sorted(weights.items())])
            cos_sim *= weights

        indices = np.argsort(-cos_sim)  # sort in descending order
        dt_index = -1
        for index in indices:
            candidate_table = list_candidate_source_tables[index]
            list_source_floats = extract_floats_from_table(candidate_table)
            list_shared_floats = find_shared_floats(list1=list_target_floats, list2=list_source_floats)

            if len(list_shared_floats) > 1 and cos_sim[index] > self.threshold:
                dt_index = index
                break

        dt_source_table = list_candidate_source_tables[dt_index] if dt_index != -1 else ''
        return dt_source_table, None

    def __call__(
            self,
            target_arxiv_id: str,
            source_arxiv_id: str,
    ):
        target_table_to_embedding, gt_target_table, gt_source_table = self.get_table_to_embedding(
            target_arxiv_id=target_arxiv_id, source_arxiv_id=source_arxiv_id, is_target=True
        )
        target_embeddings: np.ndarray = np.array(list(target_table_to_embedding.values()))

        source_table_to_embedding = self.get_table_to_embedding(
            target_arxiv_id=target_arxiv_id, source_arxiv_id=source_arxiv_id, is_target=False
        )
        source_embeddings: np.ndarray = np.array(list(source_table_to_embedding.values()))

        dt_source_table, response = self.match(
            target_embeddings=target_embeddings,
            source_embeddings=source_embeddings,
            target_table=gt_target_table,
            list_candidate_source_tables=list(source_table_to_embedding.keys())
        )
        return gt_source_table.replace('\n', ''), dt_source_table.replace('\n', ''), response


if __name__ == '__main__':
    from argparse import ArgumentParser
    from glob import glob
    from tqdm import tqdm
    from utils.table_matching_evaluator import TableMatchingEvaluator

    parser = ArgumentParser()
    parser.add_argument(
        "--dir_root",
        type=str,
        default="/Users/noel/projects/caml/research/arxiveri"
    )
    parser.add_argument(
        "--dir_table_to_embedding",
        type=str,
        default="table_to_embedding"
    )
    parser.add_argument(
        "--embedding_model",
        "-e",
        type=str,
        default="ada-002",
        choices=["ada-002", "embed-english-light-v2.0", "embed-english-v2.0", "embed-multilingual-v2.0"],
    )
    parser.add_argument("--format", '-f', type=str, default="html", choices=["html", "csv", "markdown"])
    parser.add_argument(
        "--dir_table_match",
        type=str,
        default="dataset/ground_truth/table_match"
    )
    parser.add_argument(
        "--dir_source_papers",
        type=str,
        default="dataset/source_papers"
    )
    parser.add_argument(
        "--dir_ckpt",
        type=str,
        default="results/table_match"
    )
    parser.add_argument("--weighting", "-w", action="store_true", default=False)
    args = parser.parse_args()

    args.dir_table_to_embedding = f"{args.dir_root}/{args.dir_table_to_embedding}"
    args.dir_table_match = f"{args.dir_root}/{args.dir_table_match}"
    args.dir_source_papers = f"{args.dir_root}/{args.dir_source_papers}"

    dir_ckpt = f"{args.dir_root}/{args.dir_ckpt}/{args.embedding_model.replace('-', '_')}_format_{args.format}"

    # count the number of directories that start with "run_" in the checkpoint directory
    list_run_dir: List[str] = sorted(glob(f"{dir_ckpt}/run_*"))
    n_previous_runs: int = len(list_run_dir)
    dir_ckpt = f"{dir_ckpt}/run_{n_previous_runs:02d}"
    print(f"dir_ckpt: {dir_ckpt}")

    table_matcher = TableMatcher(
        dir_table_to_embedding=args.dir_table_to_embedding,
        dir_table_match=args.dir_table_match,
        dir_source_papers=args.dir_source_papers,
        embedding_model=args.embedding_model,
        weighting=args.weighting,
        format=args.format,
    )

    table_matching_evaluator: callable = TableMatchingEvaluator()

    list_fp_table_matches: List[str] = sorted(glob(f"{args.dir_table_match}/*/*.json"))
    assert len(list_fp_table_matches) > 0

    pbar = tqdm(list_fp_table_matches)
    for fp_table_match in pbar:
        target_arxiv_id: str = fp_table_match.split("/")[-2]
        source_arxiv_id: str = fp_table_match.split("/")[-1].replace(".json", "")
        gt_source_table, dt_source_table, response = table_matcher(
            target_arxiv_id=target_arxiv_id,
            source_arxiv_id=source_arxiv_id,
        )

        table_match_accuracy: float = table_matching_evaluator(
            gt_source_table=gt_source_table, dt_source_table=dt_source_table
        )
        pbar.set_description(f"table_match_accuracy: {table_match_accuracy:.3f}")

    fp_metric: str = f"{dir_ckpt}/metric.json"
    os.makedirs(os.path.dirname(fp_metric), exist_ok=True)
    json.dump(table_matching_evaluator.get_score(), open(fp_metric, "w"))
