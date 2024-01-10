import unittest

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from promptengineers.llms.services.langchain.callbacks import AgentStreamCallbackHandler, AsyncIteratorCallbackHandler
from promptengineers.llms.strategies import ModelContext, OllamaStrategy
from promptengineers.llms.services.langchain.chains import ChainService
from promptengineers.stream.utils import token_stream, end_stream
from langchain.schema import HumanMessage, SystemMessage, AIMessage


chat_history = [('Who won the 2001 world series?', 'The arizona diamondbacks won the 2001 world series.')]
filtered_messages = ['Who won the 2001 world series?', 'The arizona diamondbacks won the 2001 world series.', 'Who were the pitchers?']

class TestOllamaStrategy(unittest.TestCase):

	@unittest.skip("Skip testing for now, need to setup model")
	def test_ollama_query(self):
		callback = StreamingStdOutCallbackHandler()
		model_service = ModelContext(strategy=OllamaStrategy())
		llm = model_service.chat(
			model_name='llama2:7b',
			temperature=0.1,
			callbacks=[callback]
		)
		chain = ChainService(llm).conversation()
		# query = {'input': filtered_messages[-1], 'context': chat_history}
		messages = [
			# SystemMessage(
			# 	content="You are a helpful AI assitant that responds like a pirate."
			# ), ## Does not work with System Message
			HumanMessage(
				content="Who won the 2001 world series?"
			),
			AIMessage(
				content="The 2001 World Series was won by the Arizona Diamondbacks, who defeated the New York Yankees in 7 games. The Diamondbacks rallied from being down 2-0 and 3-2 in the series to win their first World Series title in just their 4th season as an expansion franchise. The MVP was pitcher Randy Johnson.content=' The 2001 World Series was won by the Arizona Diamondbacks, who defeated the New York Yankees in 7 games. The Diamondbacks rallied from being down 2-0 and 3-2 in the series to win their first World Series title in just their 4th season as an expansion franchise. The MVP was pitcher Randy Johnson."
			),
			HumanMessage(
				content="Who were the team owners at the time?"
			),
		]
		result = chain.run(filtered_messages[0], callbacks=[callback])
		print(result)