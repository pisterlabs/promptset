"""
Test the lookup_specs_chain

NOTE: this just makes sure the chain executes properly but DOES NOT assess the quality of the agent's analysis. That is done in the ipython notebooks in the evals/ folder
"""
import pytest
import json
from meche_copilot.get_equipment_results import get_spec_lookup_data, get_spec_page_validations
from meche_copilot.chains.analyze_specs_chain import AnalyzeSpecsChain
from meche_copilot.chains.helpers.specs_retriever import SpecsRetriever
from meche_copilot.schemas import *
from meche_copilot.utils.chunk_dataframe import chunk_dataframe
from meche_copilot.utils.config import load_config, find_config

# Used for verbose output of langchain prompts and responses
from langchain.callbacks import StdOutCallbackHandler

@pytest.mark.skip(reason="TODO - desiding if I want to use analyze specs chain or lookup specs chain...probably analyze specs chain")
def test_lookup_specs_chain(session:Session):
    sess = session
    config = session.config

    # a piece of equipment (idx 2: fan eq)
    eq = sess.equipments[2]

    # empty_eq_str_srcA, empty_eq_str_srcB = get_spec_lookup_data(eq)

    empty_eq_df = eq.instances_to(ScopedEquipment.IOFormats.df)
    spec_defs_df = eq.spec_defs_to(ScopedEquipment.IOFormats.df)

    concated = pd.concat([spec_defs_df.T, empty_eq_df])

    retriever = SpecsRetriever(doc_retriever=config.doc_retriever, source=eq.sourceA)
    relavent_docs = retriever.get_relevant_documents(query="")
    relavent_ref_docs_as_dicts = [doc.dict() for doc in relavent_docs]
    relavent_ref_docs_as_string = json.dumps(relavent_ref_docs_as_dicts)  # Convert to JSON string

    # lookup specs for source A
    lookup_chain = AnalyzeSpecsChain(
        doc_retriever=sess.config.doc_retriever,
        spec_reader=sess.config.spec_reader,
        callbacks=[StdOutCallbackHandler()]
    )

    result_sourceA = lookup_chain.run({
        "source": eq.sourceA,
        # "refresh_source_docs": False
        # "spec_def_df": spec_defs_df,
        "spec_res_df": concated,
    })

    from langchain.schema import AIMessage, HumanMessage, SystemMessage
    messages = [
        SystemMessage(
            content=f"For each key in results_json, find the corresponding spec in the Context using the definition and replace 'None' with correct value. Context: {relavent_ref_docs_as_string}"
        ),
        HumanMessage(
            content=f"results_json={concated.iloc[:, 0:5].to_json()}"
        ),
    ]
    lookup_chain.chat(messages)

    result_sourceA

    result_sourceA_validated = get_spec_page_validations(val_pg=result_sourceA, ref_docs=eq.sourceA.ref_docs)