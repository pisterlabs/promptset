from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import os.path
import time
from chromadb.config import Settings
from profiles import *
from utils import *

class AIDE :

	def __init__(self, profile='main', model='main', db='main', mute_stream=False, root_dir='./'):

		self.profile_name = profile
		self.model_name = model
		self.db_name = db

		slash = '' if root_dir.endswith('/') else '/'
		self.profiles = Profiles(f'{root_dir}{slash}profiles')
		self.profiles.load_profile(profile)
		
		self.p = self.profiles.profile_alias(profile)
		self.m = self.p.model_alias(model)
		# print(self.p)
		log(f'Profile: {profile}')
		log(f'Model: {model}')
		log(f'DB: {db}')

		self.embeddings = HuggingFaceEmbeddings(model_name=self.m['embeddings_model'])
		self.setup_db(db_name=db)

		# activate/deactivate the streaming StdOut callback for LLMs
		self.callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
		self.setup_model()
		self.setup_qa()


	def setup_db(self, db_name):
		db_path = self.p.db_path(db_name)
		settings = Settings(**self.p.dbs[db_name]['client_settings'])
		settings.persist_directory = db_path #because of langchain bug


		self.db = Chroma(persist_directory=db_path, embedding_function=self.embeddings, client_settings=settings)
		self.retriever = self.db.as_retriever(search_kwargs={"k": self.m['target_source_chunks']})

	def setup_qa(self):
		self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever, return_source_documents=True)

	def switch_db(self, db_name):
		log(f'Switching to DB : {db_name}')
		if not self.p.db_exists(db_name):
			err(f'DB does not exists : {db_name}')
			return
		self.db_name = db_name
		self.setup_db(db_name)
		self.setup_qa()

	def switch_model(self, model_name):
		log(f'Switching to Model : {model_name}')
		if model_name not in self.p.models :
			err(f"Model config does not exists : {model_name}")
			return

		self.model_name = model_name
		self.m = self.p.model_alias(self.model_name)
		del self.llm
		self.setup_model()


	def setup_model(self): # Prepare the LLM
		params = { k:v for k,v in self.m.items() if k not in ['type', 'embeddings_model', 'target_source_chunks']}
		print(params)
		params['callbacks'] = self.callbacks

		if not os.path.isfile(params['model_path']) :
			err(f"Setup model: Model does not exists : {params['model_path']}")
			return

		match self.m['type']:
			case "LlamaCpp":
				self.llm = LlamaCpp(**params)
				# self.llm = LlamaCpp(model_path=self.m['path'], n_ctx=self.m['n_ctx'], n_batch=self.m['n_batch'], callbacks=self.callbacks, verbose=False, n_gpu_layers=self.m['n_gpu_layers'])
			case "GPT4All":
				self.llm = GPT4All(**params)
				self.llm = GPT4All(model=self.m['path'], n_ctx=self.m['n_ctx'], backend='gptj', n_batch=self.m['n_batch'], callbacks=self.callbacks, verbose=False)
			case _default:
				# raise exception if model_type is not supported
				raise Exception(f"Model type {self.m['type']} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
	

	def step(self, query, mode='qa'):

		# Get the answer from the chain
		start = time.time()
		res = {}
		if mode == 'direct' : res['result'] = self.llm(query)
		else : res = self.qa(query)
		end = time.time()
		total_time = round(end - start, 2)

		res['time'] = total_time
		return res

