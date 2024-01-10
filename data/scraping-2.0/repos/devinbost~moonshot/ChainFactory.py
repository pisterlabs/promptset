import json
from operator import itemgetter
from typing import List, Dict, Any

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, Runnable

import PromptFactory
from DataAccess import DataAccess
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo


class ChainFactory:
    def __new__(cls, *args, **kwargs):
        print("1. Create a new instance of Point.")
        return super().__new__(cls)

    @staticmethod
    def build_summarization_chain(
        model: ChatOpenAI, data_access: DataAccess, table_schema: TableSchema
    ) -> Runnable:
        """
        Builds a chain for summarizing table data using a specified model and data access object.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            data_access (DataAccess): The data access object for executing queries.
            table_schema (TableSchema): The schema of the table to be summarized.
        Returns:
            Runnable: A runnable chain for table data summarization.
        """

        def test_func(testme: Dict[str, Any]) -> Any:
            print(testme)
            print(table_schema)
            user_props = testme["user_info_not_summary"]
            return user_props

        top3_chain: Runnable = (
            PromptFactory.build_select_query_for_top_three_rows_parallelizable(
                table_schema
            )
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(data_access.exec_cql_query_simple)
        )
        select_with_where_chain: Runnable = (
            {
                "Top3Rows": top3_chain,
                "PropertyInfo": test_func,  # Should be user properties of some kind
            }
            | PromptFactory.build_select_query_with_where_parallelizable(table_schema)
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(data_access.exec_cql_query_simple)
        )
        table_summarization_chain: Runnable = (
            {"Information": select_with_where_chain}
            | PromptFactory.build_summarization_prompt()
            | model
            | StrOutputParser()
        )
        return table_summarization_chain

    def build_summarization_chain_set(
        self, model: ChatOpenAI, data_access: DataAccess, tables: List[TableSchema]
    ) -> RunnableParallel:
        """
        Builds a set of summarization chains for a list of table schemas.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            data_access (DataAccess): The data access object for executing queries.
            tables (List[TableSchema]): A list of table schemas to be summarized.
        Returns:
            RunnableParallel: A parallel runnable chain for summarizing multiple tables.
        """
        chains = [
            self.build_summarization_chain(model, data_access, table)
            for table in tables
        ]
        chain_kwargs = {f"chain{i+1}": chain for i, chain in enumerate(chains)}
        # Pass the dictionary as keyword arguments
        summarization_chains = RunnableParallel(**chain_kwargs)
        return summarization_chains

    def build_collection_summarization_chain(
        model: ChatOpenAI, data_access: DataAccess, table_schema: TableSchema
    ) -> Runnable:
        """
        Builds a chain for summarizing collection data using a specified model and data access object.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            data_access (DataAccess): The data access object for executing queries.
            table_schema (TableSchema): The schema of the table to be summarized.
        Returns:
            Runnable: A runnable chain for collection data summarization.
        """
        top3_chain = (
            PromptFactory.build_select_query_for_top_three_rows_parallelizable(
                table_schema
            )
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(data_access.exec_cql_query_simple)
        )
        select_with_where_chain = (
            {
                "Top3Rows": top3_chain,
                "PropertyInfo": itemgetter("property_info"),
            }
            | PromptFactory.build_select_query_with_where_parallelizable(table_schema)
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(data_access.exec_cql_query_simple)
        )
        table_summarization_chain = (
            {"Information": select_with_where_chain}
            | PromptFactory.build_summarization_prompt()
            | model
            | StrOutputParser()
        )
        return table_summarization_chain

    # def build_summarization_chains_from_list(self, table_schemas: List[TableSchema]):
    #     chains = [
    #         ChainFactory.build_summarization_chain(model, fake_data_access, table)
    #         for table in table_schemas
    #     ]

    def build_user_summarization_chain_parallelizable(
        self, data_access: DataAccess, model: ChatOpenAI, user_info: UserInfo
    ) -> RunnableParallel:
        """
        Builds a user summarization chain that is parallelizable.
        Parameters:
            data_access (DataAccess): The data access object for executing queries.
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            user_info (UserInfo): User information to be used in the summarization process.
        Returns:
            RunnableParallel: A parallelizable runnable chain for user data summarization.
        """

        def testme2(testme):
            print(testme)
            return testme["user_info_not_summary"]

        relevant_user_table_chain = (
            {
                "TableList": RunnableLambda(data_access.get_table_schemas_in_db_v2),
                "UserInfo": RunnableLambda(testme2),
            }
            | PromptFactory.build_table_identification_prompt()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(
                data_access.map_tables_and_populate
            )  # Not the most performant to rebuild column metadata here, but we can optimize later
        )
        relevant_tables = relevant_user_table_chain.invoke(
            {"user_info_not_summary": user_info}
        )
        factory = ChainFactory()
        # Pass the dictionary as keyword arguments
        user_summarization_chain_parallelizable = factory.build_summarization_chain_set(
            model, data_access, relevant_tables
        )
        return user_summarization_chain_parallelizable

    def build_path_segment_keyword_chain(
        self,
        model: ChatOpenAI,
        user_info_summary_parallelizable_chain: RunnableParallel,
        data_access: DataAccess,
    ) -> Runnable:
        """
        Builds a chain for extracting path segment keywords.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            user_info_summary_parallelizable_chain (RunnableParallel): A parallelizable chain for user information summarization.
            data_access (DataAccess): The data access object for executing queries.
        Returns:
            Runnable: A runnable chain for path segment keyword extraction that list of filters like this:
            [{{"metadata.path_segment_X": "VALUE"}}]
        """

        path_segment_keyword_chain = (
            {
                "PathSegmentValues": RunnableLambda(
                    data_access.get_path_segment_keywords
                ),
                "UserInformationSummary": user_info_summary_parallelizable_chain,
            }
            | PromptFactory.build_collection_vector_find_prompt()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
        )
        return path_segment_keyword_chain

    def build_collection_predicate_chain_non_parallel(
        self,
        model: ChatOpenAI,
        user_info_summary: Any,
        data_access: DataAccess,
    ) -> Runnable:
        """
        Builds a non-parallel chain for collection predicate extraction.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            user_info_summary (Any): The summary of user information.
            data_access (DataAccess): The data access object for executing queries.
        Returns:
            Runnable: A runnable chain for collection predicate extraction.
            that Returns list of filters like this: [{{"metadata.path_segment_X": "VALUE"}}]
        """
        path_segment_keyword_chain = (
            {
                "PathSegmentValues": RunnableLambda(
                    data_access.get_path_segment_keywords
                ),
                "UserInformationSummary": RunnableLambda(lambda x: user_info_summary),
            }
            | PromptFactory.build_collection_vector_find_prompt()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
        )
        return path_segment_keyword_chain

    def build_collection_predicate_chain_non_parallel_v2(
        self,
        model: ChatOpenAI,
        table_summarization: str,
        all_keywords: Dict[str, List[str]],
    ) -> Runnable:
        """
        Builds a non-parallel version 2 chain for collection predicate extraction.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            table_summarization (str): The summarization of table data.
            all_keywords (Dict[str, List[str]]): A dictionary of all keywords to be considered.
        Returns:
            Runnable: A runnable chain for collection predicate extraction.
            that Returns list of filters like this: [{{"metadata.path_segment_X": "VALUE"}}]
        """
        all_keywords_string: str = json.dumps(all_keywords)
        # print("All Keywords String:", all_keywords_string)
        # print("Table Summarization:", table_summarization)

        path_segment_values_lambda = RunnableLambda(lambda x: all_keywords_string)
        user_information_summary_lambda = RunnableLambda(lambda x: table_summarization)

        # print("Path Segment Values Lambda:", path_segment_values_lambda)
        # print("User Information Summary Lambda:", user_information_summary_lambda)
        #
        # prompt_factory_result = PromptFactory.build_collection_vector_find_prompt_v2()
        # print("Prompt Factory Result:", prompt_factory_result)
        path_segment_keyword_chain = (
            {
                "PathSegmentValues": path_segment_values_lambda,
                "UserInformationSummary": user_information_summary_lambda,
            }
            | PromptFactory.build_collection_vector_find_prompt_v4()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(lambda x: json.loads(x))
        )
        return path_segment_keyword_chain

    def build_vector_search_summarization_chain(
        self, model: ChatOpenAI, search_results: str
    ) -> Runnable:
        """
        Builds a chain for summarizing vector search results.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            search_results (str): The search results to be summarized.
        Returns:
            Runnable: A runnable chain for vector search result summarization.
        """
        collection_summary_chain = (
            {"Information": RunnableLambda(lambda x: search_results)}
            | PromptFactory.build_summarization_prompt()
            | model
            | StrOutputParser()
        )
        return collection_summary_chain

    def build_astrapy_collection_summarization_chain(
        self,
        model: ChatOpenAI,
        user_info_summary_parallelizable_chain: RunnableParallel,
        data_access: DataAccess,
    ) -> Runnable:
        """
        Builds a chain for summarizing AstraPy collection data.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            user_info_summary_parallelizable_chain (RunnableParallel): A parallelizable chain for user information summarization.
            data_access (DataAccess): The data access object for executing queries.
        Returns:
            Runnable: A runnable chain for AstraPy collection data summarization.
        """

        def run_query(filter_and_summary):
            collection_filter = filter_and_summary["filter"]
            user_summary = filter_and_summary["user_summary"]
            search_results = data_access.filtered_ANN_search(
                collection_filter, user_summary
            )
            return search_results

        path_segment_keyword_chain = (
            {
                "filters": {
                    "PathSegmentValues": RunnableLambda(
                        data_access.get_path_segment_keywords
                    ),
                    "UserInformationSummary": user_info_summary_parallelizable_chain,
                }
                | PromptFactory.build_collection_vector_find_prompt()
                | model
                | StrOutputParser()
                | RunnableLambda(PromptFactory.clean_string_v2)
            }
            | PromptFactory.build_filter_and_summary_joiner_prompt()
            | model
            | RunnableLambda(PromptFactory.clean_string_v2)
            | StrOutputParser()
        )
        return path_segment_keyword_chain

    def build_astrapy_collection_summarization_chain_v2(
        self,
        model: ChatOpenAI,
        user_info_summary: Any,
        data_access: DataAccess,
    ) -> Runnable:
        """
        Builds a version 2 chain for summarizing AstraPy collection data.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            user_info_summary (Any): The summary of user information.
            data_access (DataAccess): The data access object for executing queries.
        Returns:
            Runnable: A runnable chain for AstraPy collection data summarization.
        """

        path_segment_keyword_chain = (
            {
                "filters": {
                    "PathSegmentValues": RunnableLambda(
                        data_access.get_path_segment_keywords
                    ),
                    "UserInformationSummary": user_info_summary,
                }
                | PromptFactory.build_collection_vector_find_prompt()
                | model
                | StrOutputParser()
                | RunnableLambda(PromptFactory.clean_string_v2)
            }
            | PromptFactory.build_filter_and_summary_joiner_prompt()
            | model
            | RunnableLambda(PromptFactory.clean_string_v2)
            | StrOutputParser()
        )
        return path_segment_keyword_chain

    def build_astrapy_collection_summarization_prep_chain(
        self,
        model: ChatOpenAI,
        user_info_summary_parallelizable_chain: RunnableParallel,
        data_access: DataAccess,
    ) -> Runnable:
        """
        Builds a preparatory chain for summarizing AstraPy collection data.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            user_info_summary_parallelizable_chain (RunnableParallel): A parallelizable chain for user information summarization.
            data_access (DataAccess): The data access object for executing queries.
        Returns:
            Runnable: A runnable chain for AstraPy collection data summarization preparation.
        """

        path_segment_keyword_chain = (
            {
                "filters": {
                    "PathSegmentValues": RunnableLambda(
                        data_access.get_path_segment_keywords
                    ),
                    "UserInformationSummary": user_info_summary_parallelizable_chain,
                }
                | PromptFactory.build_collection_vector_find_prompt()
                | model
                | StrOutputParser()
                | RunnableLambda(PromptFactory.clean_string_v2)
            }
            | PromptFactory.build_filter_and_summary_joiner_prompt()
            | model
            | RunnableLambda(PromptFactory.clean_string_v2)
            | StrOutputParser()
        )
        return path_segment_keyword_chain

    def build_final_recommendation_chain(
        self,
        user_summarization_chain_parallelizable: RunnableParallel,
        collection_summarization_chain_parallelizable: RunnableParallel,
        model: ChatOpenAI,
    ) -> Runnable:
        """
        Builds a final recommendation chain combining user and collection summarization chains.
        Parameters:
            user_summarization_chain_parallelizable (RunnableParallel): A parallelizable chain for user summarization.
            collection_summarization_chain_parallelizable (RunnableParallel): A parallelizable chain for collection summarization.
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
        Returns:
            Runnable: A runnable chain for final recommendation.
        """
        final_chain = (
            {
                "UserSummary": user_summarization_chain_parallelizable,
                "BusinessSummary": collection_summarization_chain_parallelizable,
                "UserMessages": itemgetter("user_messages"),
                "OurResponses": itemgetter("our_responses")
                | RunnableLambda(lambda x: x.reverse()),
            }
            | PromptFactory.build_final_response_prompt()
            | model
            | StrOutputParser()
        )
        return final_chain

    def build_final_recommendation_chain_non_parallel(
        self,
        model: ChatOpenAI,
    ):
        final_chain = (
            {
                "UserSummary": itemgetter("user_summary"),
                "BusinessSummary": itemgetter("business_summary"),
                "UserMessages": itemgetter("user_messages"),
                "OurResponses": itemgetter("our_responses")
                | RunnableLambda(lambda x: x.reverse()),
            }
            | PromptFactory.build_final_response_prompt()
            | model
            | StrOutputParser()
        )
        return final_chain
