"""
Test the analyze_specs_chain

NOTE: this just makes sure the chain executes properly but DOES NOT assess the quality of the agent's analysis. That is done in the ipython notebooks in the evals/ folder

"""
import pytest
from meche_copilot.schemas import Session
from meche_copilot.chains.read_design_chain import ReadDesignChain
from meche_copilot.chains.read_submittal_chain import ReadSubmittalChain
from meche_copilot.chains.analyze_specs_chain import AnalyzeSpecsChain

# Used for verbose output of langchain prompts and responses
# from langchain.callbacks import StdOutCallbackHandler

@pytest.mark.skip(reason="TODO - write when done")
def test_read_design_chain(session: Session):
    read_design_chain = ReadDesignChain()
    eqs_with_design_data = read_design_chain({'equipments': session.equipments})
    assert eqs_with_design_data is not None

@pytest.mark.skip(reason="TODO - write when done")
def test_read_submittal_chain(session: Session):
    # read design chain to get equipment with design data
    read_design_chain = ReadDesignChain()
    eqs_with_design_data = read_design_chain({'equipments': session.equipments})

    # run read submittal chain to get equipment with submittal data
    read_submittal_chain = ReadSubmittalChain()
    eqs_with_submittal_data = read_submittal_chain({'equipments': eqs_with_design_data})
    assert eqs_with_submittal_data is not None

@pytest.mark.skip(reason="TODO - deciding if I want to use analyze specs chain or lookup specs chain...probably analyze specs chain")
def test_analyze_specs_chain(session: Session):
    # read design chain to get equipment with design data
    read_design_chain = ReadDesignChain()
    eqs_with_design_data = read_design_chain({'equipments': session.equipments})

    # run read submittal chain to get equipment with submittal data
    read_submittal_chain = ReadSubmittalChain()
    eqs_with_submittal_data = read_submittal_chain({'equipments': eqs_with_design_data})

    # run analyze specs chain
    analyze_specs_chain = AnalyzeSpecsChain()
    analyze_specs_chain.get_spec_results_for_eq_instance()
    analyze_specs_chain.analyze_spec_results_for_eq_instance()
    eqs_with_analysis = analyze_specs_chain({'equipments': eqs_with_submittal_data})
    assert eqs_with_analysis is not None
