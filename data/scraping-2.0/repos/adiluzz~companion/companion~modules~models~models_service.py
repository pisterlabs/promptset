import os
from langchain.chains import LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.llms import LlamaCpp
from langchain.tools import Tool, tool, ShellTool
from langchain.agents import load_tools, initialize_agent
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema.output import LLMResult
from uuid import UUID
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Any, Optional, Union
from companion.modules.chains.chains_model import Chain
from datetime import date, datetime



class MyCustomHandler(BaseCallbackHandler):
	def __init__(self, chain_id) -> None:
		super().__init__()
		self.chain_id = chain_id
 	
	def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID = None, **kwargs: Any) -> Any:
		Chain().objects(id=self.chain_id)[0].update(finished=datetime.now())
	
	def on_llm_new_token(self, token: str, **kwargs) -> None:
		chain = Chain.objects(id=self.chain_id)[0]
		if 'chain' in chain:
			new_chain = chain.chain + token
			Chain.objects(id=self.chain_id)[0].update(chain=new_chain)
		else:
			Chain.objects(id=self.chain_id)[0].update(chain=token)
	def on_llm_error(self, error: Exception | KeyboardInterrupt, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
		Chain().objects(id=self.chain_id).update(error={"error_time":datetime.now(), "error_text":error})


n_gpu_layers = 1
n_batch = 4096
n_ctx = 4096
tokens = 10000000

def get_callback_manager(chain_id):
	return CallbackManager([
		StreamingStdOutCallbackHandler(), 
		MyCustomHandler(chain_id=chain_id)
	])

def get_llm(chain_id):
	callback_manager = get_callback_manager(chain_id)
	path = os.environ['MODEL_PATH']
	llm =LlamaCpp(
		model_path=path,
		# n_gpu_layers=n_gpu_layers,
		# n_batch=n_batch,
		n_ctx=n_ctx,
		f16_kv=True,
		temperature=0.75,
		max_tokens=tokens,
		callback_manager=callback_manager,
		verbose=True,
	)
	return llm

def get_tools(llm):
	search = GoogleSearchAPIWrapper()

	def top10_results(query):
		return search.results(query, 10)

	tool = Tool(
		name="Google Search Snippets",
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
		func=top10_results,
	)
	shell_tool = ShellTool()

	tools = load_tools([
	#	'python_repl',
		'requests_all',
		'terminal',
		'wikipedia',
		'human'
	], llm=llm)
	tools.append(tool)
	tools.append(shell_tool)
	return tools

def get_agent(tools, llm, export_to_csv):
	if export_to_csv == True:
		return create_csv_agent(
			llm,
			path='./temp/export.csv',
			verbose=True,
			# agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
		)
	else:
		return initialize_agent(
			tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def run_chain(questions, prompt, chain_id):
	llm = get_llm(chain_id=chain_id)
	tools = get_tools(llm)
	agent = get_agent(tools=tools, llm=llm, export_to_csv=False)
	llm_chain = LLMChain(llm=llm, prompt=prompt)
	llm_chain.apply(questions)
	first_output = agent.run(llm_chain)
	return first_output

