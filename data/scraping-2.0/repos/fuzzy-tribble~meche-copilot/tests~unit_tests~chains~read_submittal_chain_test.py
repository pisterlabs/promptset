"""
Test read submittal chain

NOTE: this just makes sure the chain executes properly but DOES NOT assess the quality of the agent's analysis. That is done in the ipython notebooks in the evals/ folder
"""
import pytest
from meche_copilot.schemas import Session
from meche_copilot.chains.read_submittal_chain import ReadSubmittalChain

# Used for verbose output of langchain prompts and responses
# from langchain.callbacks import StdOutCallbackHandler

@pytest.fixture(scope="session")
def read_submittal_chain():
    return ReadSubmittalChain()

@pytest.mark.skip(reason="TODO - write when done")
def test_read_submittal_data(session: Session, read_submittal_chain: ReadSubmittalChain, visualize: bool):
    chain = read_submittal_chain
    chain.read_submittal_data(scoped_eq=session.equipments, show_your_work=visualize)

@pytest.mark.skip(reason="TODO - write when done")
def test_read_submittal_data_with_scoped_eq(session: Session, read_submittal_chain: ReadSubmittalChain, visualize: bool):
    """
    Test can read submittal data for a specific equipment (e.g. pump)
    """
    chain = read_submittal_chain
    pump_eq = session.equipments[0]
    pump_eq.instances[0].design_uid = 'P-1A'
    chain.read_submittal_data(scoped_eq=[pump_eq], show_your_work=visualize)
    res = chain(scoped_eq=session.equipments)


