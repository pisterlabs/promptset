import dotenv
import logging
import sys

from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from llama_index.tools import QueryEngineTool, ToolMetadata

from secIndex import get_edgar_index
from utils import get_env_var

def create_filing_tool(cik: str, company_name: str):
	index = get_edgar_index(cik)
	print(f"Loaded index and tool for {company_name}")

	# create the chat engine
	company_filing_tool = QueryEngineTool(
		query_engine=index.as_query_engine(),
		metadata=ToolMetadata(
			name= f"company_filing_tool_{company_name}",
			description= "A tool for querying {company_name} SEC filings"
		),
	)
	return company_filing_tool

if __name__ == "__main__":
	# Set up env vars
	dotenv.load_dotenv()
	open_api_key = get_env_var('OPENAI_API_KEY')

	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	logging.getLogger("pyrate_limiter").setLevel(logging.WARNING)

	logging.getLogger().handlers.clear()
	logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

	msft_tool = create_filing_tool("0000789019", "Microsoft")
	amzn_tool = create_filing_tool("0001018724", "Amazon")

	llm = OpenAI(model="gpt-3.5-turbo-0613")
	agent = ReActAgent.from_tools([msft_tool, amzn_tool], llm=llm, verbose=True)

	
	print(f"Ask me questions about Microsoft and Amazon!")
	agent.chat_repl()

