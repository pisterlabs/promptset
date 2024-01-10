from pprint import pprint
from typing import Tuple

from decouple import config

from python.embeddings.ann_search import AnnSearch
from python.embeddings.embedding import generate_embedding
from python.embeddings.openai_embedder import OpenAIEmbedder
from python.sql.types import DbElementIds, ElementIdsAndScores, ElementScores


class TrainingRanker:
    rankings: ElementIdsAndScores

    def __init__(self, datasource_id: int, match_limit: int = 1000_000, value_hint_search_limit: int = 5000):
        self.match_limit = match_limit
        self.value_hint_search_limit = value_hint_search_limit

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
        self, query: str, embedder=OpenAIEmbedder, cache_results=True
    ) -> Tuple[ElementIdsAndScores, ElementIdsAndScores, ElementIdsAndScores]:
        query_embedding = generate_embedding(query, embedder=embedder, cache_results=cache_results)

        # Fetch ranked table id
        table_name_matches = self.idx_table_name.search(query_embedding, self.match_limit)
        table_and_all_column_names_matches = self.idx_table_and_all_column_names.search(
            query_embedding, self.match_limit
        )
        table_and_all_column_names_and_all_values_matches = self.idx_table_and_all_column_names_and_all_values.search(
            query_embedding, self.match_limit
        )
        # log.debug("Table matches", matches=table_matches)

        column_name_matches = self.idx_column_name.search(query_embedding, self.match_limit)
        table_and_column_name_matches = self.idx_table_and_column_name.search(query_embedding, self.match_limit)
        column_name_and_all_values_matches = self.idx_column_name_and_all_values.search(
            query_embedding, self.match_limit
        )
        table_and_column_name_and_all_values_matches = self.idx_table_and_column_name_and_all_values.search(
            query_embedding, self.match_limit
        )

        # log.debug("Column matches", matches=columns_matches)

        # search for value hint matches in the faaise index
        value_matches = self.idx_value.search(query_embedding, self.value_hint_search_limit)
        table_column_and_value_matches = self.idx_table_column_and_value.search(
            query_embedding, self.value_hint_search_limit
        )
        # print(
        #     "Query Match Len: ",
        #     len(idx_table_names_rankings),
        #     len(idx_column_names_rankings),
        #     len(idx_table_and_column_names_rankings),
        #     len(idx_column_name_and_all_column_values_rankings),
        #     len(idx_table_column_and_value_rankings),
        #     len(idx_values_rankings),
        # )

        table_scores: ElementIdsAndScores = {}
        column_scores: ElementIdsAndScores = {}
        value_scores: ElementIdsAndScores = {}

        self.merge_search_results(table_scores, table_name_matches, "table_name")
        self.merge_search_results(table_scores, table_and_all_column_names_matches, "table_and_all_column_names")
        self.merge_search_results(
            table_scores, table_and_all_column_names_and_all_values_matches, "table_and_all_column_names_and_all_values"
        )

        # When merging columns, pass in the table scores dict as well so we can look up the table scores and copy
        # it over.
        self.merge_search_results(column_scores, column_name_matches, "column_name", table_scores)
        self.merge_search_results(column_scores, table_and_column_name_matches, "table_and_column_name", table_scores)
        self.merge_search_results(
            column_scores, column_name_and_all_values_matches, "column_name_and_all_values", table_scores
        )
        self.merge_search_results(
            column_scores,
            table_and_column_name_and_all_values_matches,
            "table_and_column_name_and_all_values",
            table_scores,
        )

        self.merge_search_results(value_scores, value_matches, "value")
        self.merge_search_results(value_scores, table_column_and_value_matches, "table_column_and_value")

        return table_scores, column_scores, value_scores

    def merge_search_results(
        self,
        scores: ElementIdsAndScores,
        faiss_rankings: list[tuple[float, DbElementIds]],
        ranking_name: str,
        table_scores: ElementIdsAndScores | None = None,
    ):
        for ranking in faiss_rankings:
            score, db_element = ranking

            if db_element not in scores:
                scores[db_element] = ElementScores()

            current_scores = scores[db_element]

            # Update the score on the ElementRanking
            setattr(current_scores, ranking_name + "_score", score)

            if table_scores:
                # For columns, we pass in the table_scores dict as well, and use it to find the table scores for the
                # table and append those to the column scores
                lookup = DbElementIds(db_element[0], None, None)

                if lookup in table_scores:
                    # Copy the table related scores from the table only scores
                    table_score = table_scores[lookup]

                    current_scores.table_name_score = table_score.table_name_score
                    current_scores.table_and_all_column_names_score = table_score.table_and_all_column_names_score
                    current_scores.table_and_all_column_names_and_all_values_score = (
                        table_score.table_and_all_column_names_and_all_values_score
                    )


if __name__ == "__main__":
    ranker = TrainingRanker(1)

    rank_results = ranker.rank("How many orders from Montana?")
    pprint(rank_results)
