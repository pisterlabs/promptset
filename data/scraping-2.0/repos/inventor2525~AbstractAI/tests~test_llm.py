import openai
from AbstractAI.LLMs.StableBeluga2 import StableBeluga2
from AbstractAI.LLMs.OpenAI_LLM import OpenAI_LLM, LLM
from AbstractAI.Conversation import Conversation, Message, Role
import unittest

run_tests = False

class TestLLM(unittest.TestCase):
	def _run_test_on(self, llm:LLM):
		# Create a conversation
		conversation = Conversation()
		conversation.add_message(Message("You are a chat bot.", Role.System))
		conversation.add_message(Message("Write me a python script to count to 5.", Role.User))

		response = llm.prompt(conversation)
		conversation.add_message(response.message)
		self.assertEqual(len(conversation.message_sequence.messages), 3)
		self.assertGreater(len(response.message.content), 0)

		# Add another user message
		conversation.add_message(Message("Not like that.", Role.User))

		# Prompt the model again
		response = llm.prompt(conversation)
		conversation.add_message(response.message)
		self.assertEqual(len(conversation.message_sequence.messages), 5)
		self.assertGreater(len(response.message.content), 0)

		# Print the conversation messages
		for message in conversation.message_sequence.messages:
			print(f"{message.role}: {message.content}")
	
	def test_openai_llm_conversation(self):
		if run_tests:
			llm = OpenAI_LLM()
			llm.start()
			
			self._run_test_on(llm)
			
	def test_stable_beluga2_conversation(self):
		if run_tests:
			llm = StableBeluga2("stabilityai/StableBeluga-7B")
			llm.start()
			
			self._run_test_on(llm)

if __name__ == '__main__' and run_tests:
	unittest.main()