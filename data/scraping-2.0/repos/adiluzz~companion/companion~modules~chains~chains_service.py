from threading import Thread
from langchain.prompts import PromptTemplate
import os
from companion.modules.models.models_service import run_chain as run_chain_service
from companion.modules.chains.chains_model import Chain

class ChainsService:

	def run_chain(chain_data, title):
		template = """
  			Question: {question}. 
			If you don't know, search the internet or ask a human
     		"""
		prompt = PromptTemplate(
			template=template, input_variables=["question"])
		created_chain = Chain()
		created_chain.title = title
		created_chain.save()
		thread = Thread(target=run_chain_service, args=(chain_data, prompt, created_chain.id))
		thread.start()
		return str(created_chain.id)


