from pytest import fixture
from chains.recording_chain import RecordingChain
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms.fake import FakeListLLM


def test_create_llm_recording_chain():
  chain = RecordingChain(chain_spec_id=0, chain=LLMChain(
    prompt=PromptTemplate(input_variables=["input1", "input2"], template="prompt1 {input1} {input2}"),
    llm=FakeListLLM(responses=["fake_response1"]),
    output_key="output1",
  ))
  assert chain.input_keys == ["input1", "input2"]
  assert chain.output_keys == ["output1"]


def test_run_llm_recording_chain():
  chain = RecordingChain(chain_spec_id=0, chain=LLMChain(
    prompt=PromptTemplate(input_variables=["input1", "input2"], template="prompt1 {input1} {input2}"),
    llm=FakeListLLM(responses=["fake_response1"]),
    output_key="output1",
  ))
  assert chain.run({"input1": "input1", "input2": "input2"}) == "fake_response1"
  assert chain.calls == [(dict(input1="input1", input2="input2"), {"output1": "fake_response1"})]

  assert chain.recorded_calls == [(dict(input1="input1", input2="input2"), dict(output1="fake_response1"))]


def test_create_sequential_recording_chain():
  llm_chain = RecordingChain(chain_spec_id=0, chain=LLMChain(
    prompt=PromptTemplate(input_variables=["input1", "input2"], template="prompt1 {input1} {input2}"),
    llm=FakeListLLM(responses=["fake_response1"]),
    output_key="output1",
  ))
  chain = RecordingChain(chain_spec_id=0, chain=SequentialChain(chains=[llm_chain],
                         input_variables=["input1", "input2"]))
  assert chain.input_keys == ["input1", "input2"]
  assert chain.output_keys == ["output1"]


def test_run_sequential_recording_chain():
  llm_chain = LLMChain(
    prompt=PromptTemplate(input_variables=["input1", "input2"], template="prompt1 {input1} {input2}"),
    llm=FakeListLLM(responses=["fake_response1"]),
    output_key="output1",
  )
  chain = RecordingChain(chain_spec_id=0, chain=SequentialChain(chains=[llm_chain],
                         input_variables=["input1", "input2"]))
  assert chain.run({"input1": "input1", "input2": "input2"}) == "fake_response1"
  assert chain.calls == [({"input1": "input1", "input2": "input2"}, {"output1": "fake_response1"})]

  assert chain.recorded_calls == [({"input1": "input1", "input2": "input2"}, {"output1": "fake_response1"})]
