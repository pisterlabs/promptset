'''
Default configurations and values.
'''

class models:
	'''
	Static list of models supported by different official providers so they can be
	dynamically imported and cut on loading time and unnecessary dependencies.
	'''
	
	openai = [
		"whisper-1",
		"babbage",
		"davinci",
		"text-davinci-edit-001",
		"babbage-code-search-code",
		"text-similarity-babbage-001",
		"code-davinci-edit-001",
		"text-davinci-001",
		"ada",
		"babbage-code-search-text",
		"babbage-similarity",
		"code-search-babbage-text-001",
		"text-curie-001",
		"gpt-3.5-turbo-16k-0613",
		"code-search-babbage-code-001",
		"text-ada-001",
		"text-similarity-ada-001",
		"curie-instruct-beta",
		"ada-code-search-code",
		"ada-similarity",
		"code-search-ada-text-001",
		"text-search-ada-query-001",
		"davinci-search-document",
		"ada-code-search-text",
		"text-search-ada-doc-001",
		"davinci-instruct-beta",
		"text-similarity-curie-001",
		"code-search-ada-code-001",
		"ada-search-query",
		"text-search-davinci-query-001",
		"curie-search-query",
		"davinci-search-query",
		"babbage-search-document",
		"ada-search-document",
		"text-search-curie-query-001",
		"text-search-babbage-doc-001",
		"curie-search-document",
		"text-search-curie-doc-001",
		"babbage-search-query",
		"text-babbage-001",
		"text-search-davinci-doc-001",
		"gpt-4-0613",
		"text-embedding-ada-002",
		"text-search-babbage-query-001",
		"curie-similarity",
		"gpt-4",
		"curie",
		"gpt-4-0314",
		"text-similarity-davinci-001",
		"text-davinci-002",
		"gpt-3.5-turbo",
		"text-davinci-003",
		"davinci-similarity",
		"gpt-3.5-turbo-0301",
		"gpt-3.5-turbo-16k",
		"gpt-3.5-turbo-0613"
	]
	'''List of known models from OpenAI.'''
	
	gpt4all = [
		"gpt4all-j-v1.3-groovy",
		"gpt4all-l13b-snoozy",
		"mpt-7b-chat",
		"v3-13b-hermes-q5_1",
		"vicuna-7b-1.1-q4_2",
		"vicuna-13b-1.1-q4_2",
		"wizardLM-7B.q4_2",
		"stable-vicuna-13B.q4_2",
		"mpt-7b-base",
		"nous-gpt4-vicuna-13b",
		"mpt-7b-instruct",
		"wizard-13b-uncensored",
		"replit-code-v1-3b"
	]
	'''List of known models from GPT4All.'''

openai = dict(
	model = "gpt-3.5-turbo"
)
'''OpenAI connector defaults.'''

config = dict(
	# OpenAI
	openai_api_key = None,
	openai_organization = None,

	# GPT4All
	model_path = None,
	allow_download = True,

	# Common parameters
	model = None,

	# Generation parameters
	adapter = "chain",
	temperature = 0,
	top_p = 0.9,
	top_k = 0,
	frequency_penalty = 0,
	presence_penalty = 0,
	max_tokens = 1000,
	best_of = 1,

	# Throttling
	retry = 3,
	concurrent = 1,
	request_rate = 60,
	token_rate = 250000,
	period = 60
)
'''Default env configurations.'''

RETRY = 3
'''Default retry count for adapters.'''