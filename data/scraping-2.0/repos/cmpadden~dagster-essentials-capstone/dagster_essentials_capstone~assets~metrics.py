"""Derived metrics

References

    - https://python.langchain.com/docs/use_cases/summarization
    - https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.srt.SRTLoader.html
    - https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token
    - https://python.langchain.com/docs/modules/model_io/llms/token_usage_tracking

Next steps:

    - Determine which assets are needed by website (eg. fetch images)
    - Fix:

Notes

    If you encounter the following error when using SRTLoader, it is possible that the
    subtitles are encoded in something like ISO-8859. While `autodetect_encoding` is a
    parameter of `TextLoader`, it is not yet available in `SRTLoader`; this would be a
    nice enhancement.

        UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe7 in position 12: invalid
        continuation byte

    A temporary workaround, that should be integrated into the pipeline itself, is to
    use the `iconv` command:

         iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt



"""
import os

from dagster import asset
from dagster_duckdb import DuckDBResource
from langchain.callbacks import get_openai_callback
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SRTLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter

from dagster_essentials_capstone.constants import (
    DUCKDB_TABLE_LETTERBOXD_FILMS_DETAILS, DUCKDB_TABLE_OPENAI_SUMMARY)


@asset(deps=["film_open_subtitles_raw"])
def openai_film_summary(database: DuckDBResource):
    """ Film summary based on subtitles fed into an LLM.
    """
    with database.get_connection() as conn:
        query_results = conn.execute(
            f"""\
            select
              film_slug
            from {DUCKDB_TABLE_LETTERBOXD_FILMS_DETAILS}
            where film_slug not in (
                select
                  film_slug
                from {DUCKDB_TABLE_OPENAI_SUMMARY}
            )
            """
        ).fetchall()

    for row in query_results:
        film_slug = row[0]

        srts = [
            f
            for f in os.listdir(f"./data/staging/opensubtitles/{film_slug}/")
            if f.endswith(".srt") or f.endswith(".ass")  # ... ass?
        ]

        assert len(srts) > 0

        # TODO: Some movies, notably `the-menu-2022`, have many subtitles -- need to
        # determine which method is best for picking the primary English subtitle.
        loader = SRTLoader(f"./data/staging/opensubtitles/{film_slug}/{srts[0]}")

        docs = loader.load()

        llm = ChatOpenAI(temperature=0)

        map_template = """\
        The following is a set of documents containing the subtitles to a movie
        {docs}
        Based on this list of documents, please identify the main themes
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        reduce_template = """\
        The following is set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary of the main themes, don't
        mention the list of summaries, and only provide a final wholistic summary of the movie.
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

        text_splitter = TokenTextSplitter.from_tiktoken_encoder(
            chunk_size=3968, chunk_overlap=128
        )
        split_docs = text_splitter.split_documents(docs)

        with get_openai_callback() as cb:
            summary = map_reduce_chain.run(split_docs)

        # Example callback result:
        #
        # Tokens Used: 17515
        #         Prompt Tokens: 17140
        #         Completion Tokens: 375
        # Successful Requests: 6
        # Total Cost (USD): $0.026459999999999997

        # Example summary of "Everything Everywhere All at Once"
        #
        # The movie explores themes of family dynamics, cultural identity, communication, personal
        # growth, love and admiration, multiverse and parallel universes, personal growth and
        # self-discovery, mother-daughter relationship, and identity. It delves into the
        # complexities of relationships, the search for one's true self, and the exploration of
        # different realities. The protagonist's journey to save her daughter and the exploration
        # of various versions of themselves add depth to the narrative. Overall, the movie offers
        # a thought-provoking exploration of these themes and their impact on the characters'
        # lives.

        with database.get_connection() as conn:
            conn.execute(
                f"""\
                create table if not exists {DUCKDB_TABLE_OPENAI_SUMMARY} (
                    film_slug varchar,
                    openai_total_tokens integer,
                    openai_prompt_tokens integer,
                    openai_completion_tokens integer,
                    openai_total_cost double,
                    summary varchar
                )
                """
            )
            conn.execute(
                f"""\
                insert into {DUCKDB_TABLE_OPENAI_SUMMARY} values (?, ?, ?, ?, ?, ?)
                """,
                [
                    film_slug,
                    cb.total_tokens,
                    cb.prompt_tokens,
                    cb.completion_tokens,
                    cb.total_cost,
                    summary,
                ],
            )
