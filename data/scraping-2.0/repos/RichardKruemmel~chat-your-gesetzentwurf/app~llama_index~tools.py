import logging
from llama_index.langchain_helpers.agents.tools import LlamaIndexTool
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

from app.llama_index.index import setup_index
from app.llama_index.query_engine import setup_query_engine
from app.database.crud import get_vectorized_election_programs_from_db
from app.database.database import Session


def setup_agent_tools():
    session = Session()
    vectorized_election_programs = get_vectorized_election_programs_from_db(session)
    logging.info(f"Loaded {len(vectorized_election_programs)} vectorized programs.")

    vector_tools = []
    for program in vectorized_election_programs:
        meta_data_filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="group_id", value=program.id),
                ExactMatchFilter(key="election_id", value=program.election_id),
                ExactMatchFilter(key="party_id", value=program.party_id),
            ]
        )

        # define query engines
        vector_index = setup_index()
        vector_query_engine = setup_query_engine(
            vector_index, filters=meta_data_filters
        )
        # define tools
        query_engine_tool = LlamaIndexTool(
            name="vector_tool",
            description=(
                f"Nützlich für Fragen zu spezifischen Aspekten des Wahlprogramms der {program.full_name} für die {program.label}."
            ),
            query_engine=vector_query_engine,
        )
        logging.info(f"Loaded query engine tool for {program.full_name}.")
        vector_tools.append(query_engine_tool)
    return vector_tools
