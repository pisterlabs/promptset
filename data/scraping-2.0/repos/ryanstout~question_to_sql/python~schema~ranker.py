from python.setup import log

import os
import sys
import time
import typing as t

from decouple import config

from python.embeddings.ann_search import AnnSearch
from python.embeddings.embedding import generate_embedding
from python.embeddings.openai_embedder import OpenAIEmbedder
from python.sql.types import DbElementIds
from python.sql.utils.touch_points import convert_db_element_ids_to_db_element
from python.utils.db import application_database_connection


# Taking in the question, generate the following data structure to pass to
# the schema builder:
class ElementRank(t.NamedTuple):
    table_id: int
    column_id: t.Union[int, None]
    value_hint: t.Union[str, None]
    score: float


# note that in a ranking type list, you could have multiple ranking types with the same
# column_id and value_hints
SCHEMA_RANKING_TYPE = t.List[ElementRank]


class Ranker:
    def __init__(self, datasource_id: int):
        indexes_path = config("FAISS_INDEXES_PATH")

        # Table indexes (indexes that point to a table)
        self.idx_table_name = AnnSearch(f"{indexes_path}/{datasource_id}/table_name")
        self.idx_table_and_all_column_names = AnnSearch(f"{indexes_path}/{datasource_id}/table_and_all_column_names")
        self.idx_table_and_all_column_names_and_all_values = AnnSearch(
            f"{indexes_path}/{datasource_id}/table_and_all_column_names_and_all_values"
        )

        # Column indexes (indexes that point to a column)
        self.idx_column_name = AnnSearch(f"{indexes_path}/{datasource_id}/column_name")
        self.idx_table_and_column_name = AnnSearch(f"{indexes_path}/{datasource_id}/table_and_column_name")
        self.idx_column_name_and_all_values = AnnSearch(f"{indexes_path}/{datasource_id}/column_name_and_all_values")
        self.idx_table_and_column_name_and_all_values = AnnSearch(
            f"{indexes_path}/{datasource_id}/table_and_column_name_and_all_values"
        )

        # Cell Values (index that point to a table+column+value)
        self.idx_value = AnnSearch(f"{indexes_path}/{datasource_id}/value")
        self.idx_table_column_and_value = AnnSearch(f"{indexes_path}/{datasource_id}/table_column_and_value")

    def rank(
        self,
        query: str,
        embedder=OpenAIEmbedder,
        cache_results=True,
        table_weights=None,
        column_weights=None,
        value_weights=None,
    ) -> SCHEMA_RANKING_TYPE:
        # lists as defaults is dangerous, so we set defaults here
        # the array represents weights for 3 faiss indexes for each type
        if table_weights is None:
            table_weights = [1.5, 0.0, 0.1]
        if column_weights is None:
            column_weights = [0.1, 0.0, 0.0, 0.0]
        if value_weights is None:
            value_weights = [0.5, 0.2]

        t1 = time.time()
        query_embedding = generate_embedding(query, embedder=embedder, cache_results=cache_results)

        match_limit = 1000

        log.debug("Start ranking")
        # Fetch ranked table id
        table_name_matches = self.idx_table_name.search(query_embedding, match_limit)
        table_and_all_column_names_matches = self.idx_table_and_all_column_names.search(query_embedding, match_limit)
        table_and_all_column_names_and_all_values_matches = self.idx_table_and_all_column_names_and_all_values.search(
            query_embedding, match_limit
        )
        # log.debug("Table matches", matches=table_matches)

        column_name_matches = self.idx_column_name.search(query_embedding, match_limit)
        table_and_column_name_matches = self.idx_table_and_column_name.search(query_embedding, match_limit)
        column_name_and_all_values_matches = self.idx_column_name_and_all_values.search(query_embedding, match_limit)
        table_and_column_name_and_all_values_matches = self.idx_table_and_column_name_and_all_values.search(
            query_embedding, match_limit
        )

        # log.debug("Column matches", matches=columns_matches)

        # search for value hint matches in the faaise index
        value_hint_search_limit = 100
        value_matches = self.idx_value.search(query_embedding, value_hint_search_limit)
        table_column_and_value_matches = self.idx_table_column_and_value.search(
            query_embedding, value_hint_search_limit
        )

        tables = self.merge_ranks(
            [table_name_matches, table_and_all_column_names_matches, table_and_all_column_names_and_all_values_matches],
            table_weights,
            0,
        )
        columns = self.merge_ranks(
            [
                column_name_matches,
                table_and_column_name_matches,
                column_name_and_all_values_matches,
                table_and_column_name_and_all_values_matches,
            ],
            column_weights,
            1,
        )

        if os.getenv("DEBUG_RANKER"):
            log.debug("--------------------------------------")
            score_and_element = map(lambda x: (x[0], convert_db_element_ids_to_db_element(x[1])), value_matches)
            for score, element in score_and_element:
                log.debug("Value matches", score=score, element=element)

        values = self.merge_ranks([value_matches, table_column_and_value_matches], value_weights, 2)

        # rankings = list(map(lambda x: ElementRank(table_id=x[1][0], column_id=x[1][1], value_hint=x[1][2], score=x[0]), tables + columns + values))
        rankings: list[ElementRank] = list(
            map(
                lambda x: ElementRank(table_id=x[1][0], column_id=x[1][1], value_hint=x[1][2], score=x[0]),
                tables + columns + values,
            )
        )

        # Sort rankings by score
        rankings.sort(key=lambda x: x.score, reverse=True)

        # Print with table and column names
        if os.getenv("DEBUG_RANKER"):
            for ranking in rankings:
                db_element = convert_db_element_ids_to_db_element(
                    DbElementIds(ranking.table_id, ranking.column_id, ranking.value_hint)
                )
                log.debug("rank", score=ranking.score, element=db_element)

        t2 = time.time()
        log.debug("Ranking fetch: ", time=t2 - t1)

        return rankings

    def merge_ranks(self, scores_and_associations, weights, remove_dups_idx=None):
        # Merge the output of multiple AnnSearch#search's via the passed in
        # weights
        merged = []
        for idx, scores_and_assocs in enumerate(scores_and_associations):
            for score_and_assoc in scores_and_assocs:
                # Multiply the scores by the the associated weight
                merged.append((score_and_assoc[0] * weights[idx], score_and_assoc[1]))

        # Sort
        merged.sort(key=lambda x: x[0], reverse=True)

        if remove_dups_idx is not None:
            # Remove duplicates for the target table, column, or value
            final = []
            seen = set()
            for score_and_assoc in merged:
                if score_and_assoc[1] not in seen:
                    final.append(score_and_assoc)
                    seen.add(score_and_assoc[1])
        else:
            final = merged

        return final

    def pull_assoc(self, scores_and_assocs, assoc_idx):
        # Grabs the table, column, or value from the association tuple
        return [score_and_assoc[1][assoc_idx] for score_and_assoc in scores_and_assocs]


def dedup_schema_rankings(rankings: SCHEMA_RANKING_TYPE) -> SCHEMA_RANKING_TYPE:
    """
    Takes a schema ranking list and removes any duplicate table, column, and value hint pairs
    """
    seen = set()
    final: SCHEMA_RANKING_TYPE = []
    for ranking in rankings:
        if (ranking.table_id, ranking.column_id, ranking.value_hint) not in seen:
            final.append(ranking)
            seen.add((ranking.table_id, ranking.column_id, ranking.value_hint))

    return final


def merge_schema_rankings(rankings1: SCHEMA_RANKING_TYPE, rankings2: SCHEMA_RANKING_TYPE) -> SCHEMA_RANKING_TYPE:
    rankings = rankings1 + rankings2
    rankings.sort(key=lambda x: x.score, reverse=True)
    return dedup_schema_rankings(rankings)


if __name__ == "__main__":
    db = application_database_connection()

    datasource_id = db.datasource.find_first()
    if datasource_id:
        datasource_id = datasource_id.id
    else:
        raise Exception("No datasource found")

    question = sys.argv[1]

    ranks = Ranker(datasource_id).rank(question)
    print(ranks)
