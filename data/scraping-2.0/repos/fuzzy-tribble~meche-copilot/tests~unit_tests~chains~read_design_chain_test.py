"""
Test read design chain

NOTE: this just makes sure the chain executes properly but DOES NOT assess the quality of the agent's analysis. That is done in the ipython notebooks in the evals/ folder

NOTE: run with ...pytest tests/unit_tests/chains/read_design_chain_test.py --vis in order to see how llm and camelot are reading/interepreting the pdf data (output to data/.cache)
"""
import pytest
from meche_copilot.schemas import Session
from meche_copilot.chains.read_design_chain import ReadDesignChain

# Used for verbose output of langchain prompts and responses
# from langchain.callbacks import StdOutCallbackHandler

@pytest.fixture(scope="session")
def read_design_chain():
    return ReadDesignChain()

def test_read_design_schedules(session: Session, read_design_chain: ReadDesignChain, visualize: bool):
    chain = read_design_chain
    chain = ReadDesignChain()
    chain.read_design_schedules(scoped_eq=session.equipments, show_your_work=visualize)

@pytest.mark.skip(reason="Not implemented")
def test_read_design_drawings(session: Session, read_design_chain: ReadDesignChain, visualize: bool):
    chain = read_design_chain
    chain.read_design_drawings(scoped_eq=session.equipments, show_your_work=visualize)
