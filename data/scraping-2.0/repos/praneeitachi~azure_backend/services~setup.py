import io
import logging
import mimetypes
import os
import ast
import time
import uuid
from datetime import datetime
import aiohttp
import openai
from azure.identity.aio import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.aio import SearchClient
from azure.storage.blob.aio import BlobServiceClient
from quart import (
	Blueprint,
	abort,
	current_app,
	jsonify,
	request,
	send_file,
	send_from_directory,
)
from azure.search.documents.indexes.models import (
	SearchIndex,
	SearchField,
	SearchFieldDataType,
	SimpleField,
	SearchableField,
	SearchIndex,
	SemanticConfiguration,
	PrioritizedFields,
	SemanticField,
	SearchField,
	SemanticSettings,
	VectorSearch,
	VectorSearchAlgorithmConfiguration,
)

from utility.acshelper import get_similar_qa, qna_indexing
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.readdecomposeask import ReadDecomposeAsk
from approaches.readretrieveread import ReadRetrieveReadApproach
from approaches.retrievethenread import RetrieveThenReadApproach
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from utility.creditServiceHelper import calculate_balance, check_balance, credit_used_by_query,get_inactive_categories, update_transactions_table
load_dotenv("acs.env")


CONFIG_OPENAI_TOKEN = "openai_token"
CONFIG_CREDENTIAL = "azure_credential"
CONFIG_ASK_APPROACHES = "ask_approaches"
CONFIG_CHAT_APPROACHES = "chat_approaches"
CONFIG_BLOB_CONTAINER_CLIENT = "blob_container_client"

bp = Blueprint("routes", __name__, static_folder='static')


@bp.route("/")
async def index():
	return await bp.send_static_file("index.html")

@bp.route("/favicon.ico")
async def favicon():
	return await bp.send_static_file("favicon.ico")

@bp.route("/assets/<path:path>")
async def assets(path):
	return await send_from_directory("static/assets", path)

# Serve content files from blob storage from within the app to keep the example self-contained.
# *** NOTE *** this assumes that the content files are public, or at least that all users of the app
# can access all the files. This is also slow and memory hungry.
@bp.route("/content/<path>")
async def content_file(path):
	blob_container_client = current_app.config[CONFIG_BLOB_CONTAINER_CLIENT]
	blob = await blob_container_client.get_blob_client(path).download_blob()
	if not blob.properties or not blob.properties.has_key("content_settings"):
		abort(404)
	mime_type = blob.properties["content_settings"]["content_type"]
	if mime_type == "application/octet-stream":
		mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
	blob_file = io.BytesIO()
	await blob.readinto(blob_file)
	blob_file.seek(0)
	return await send_file(blob_file, mimetype=mime_type, as_attachment=False, attachment_filename=path)

def insert_qa_records(query, answer, start_time, thoughts, data_points,exclude_category, token_usage, email, credit_used):
	container = current_app.config['cosmos_db'].get_container_client("gi_qa")
	end_time = time.time()
	ex_time = end_time - start_time
	qa_id = uuid.uuid4()
	item = {
		'id': str(qa_id),
		'question': query,
		'answer': answer,
		'exclude_category': exclude_category,
		'feedback': '',
		'createdBy': email,
		'createdAt': str(datetime.now()),
		'updatedAt': '',
		'updatedBy': '',
		'ex_time': ex_time,
  		'thoughts':thoughts,
  		'data_points':data_points,
		'token_usage': token_usage,
		'credit_used': credit_used
	}
	container.create_item(body=item)

	return item

@bp.route("/ask", methods=["POST"])
async def ask():
	data = await request.get_json()
	email = data.get("email")
	token_usage = None
	if check_balance(email):
		try:
			print("-----------------Ask API Call happened-------------")
			start_time = time.time() 
			if not request.is_json:
				return jsonify({"error": "request must be json"}), 415
			request_json = await request.get_json()
			approach = request_json["approach"]
			query = request_json["question"]
			useai = request_json["useai"]
			exclude_category = request_json["overrides"]["exclude_category"]
			# Call get_cat_status to fetch categories with status 0
			cat_list = await get_inactive_categories()
			exclude_category.extend(cat['id'] for cat in cat_list)
			request_json["overrides"]["exclude_category"] = exclude_category
			if useai == 0:

				if exclude_category:
					exclude_category = sorted(exclude_category)
					query_string = """
					SELECT *
					FROM r
					WHERE {conditions}
						AND r.feedback = @feedback
						AND r.question = @question
					ORDER BY r.createdAt DESC
					"""
					query_params = [
						{"name": "@feedback", "value": 1},
						{"name": "@question", "value": query},
					]
					filter_conditions = " AND ".join([f'ARRAY_CONTAINS(r.exclude_category, "{i}")' for i in exclude_category])
					sql = query_string.format(conditions=filter_conditions)
				else:
					sql = "SELECT * FROM gi_qa r WHERE r.exclude_category = @exclude_category and r.feedback = 1 and r.question = '"+query+"' order by r.createdAt desc"
					query_params = [{"name": "@exclude_category", "value": exclude_category}]

				container = current_app.config['cosmos_db'].get_container_client("gi_qa")
				askResponse = {}
				query_result = container.query_items(
					query=sql,
					parameters=query_params,
					enable_cross_partition_query=True,
					max_item_count=1
				)
				qa_record = None
				for item in query_result:
					qa_record = item
					break

				if not qa_record:
					# Similar
					match_found = None
					match_found = get_similar_qa(query, exclude_category)
					if match_found == None:
						impl = current_app.config[CONFIG_ASK_APPROACHES].get(approach)
						print("-----Ask Step 1---------------")
						if not impl:
							return jsonify({"error": "unknown approach"}), 400
						# Workaround for: https://github.com/openai/openai-python/issues/371
						async with aiohttp.ClientSession() as s:
							openai.aiosession.set(s)
							print("-----Ask Step 2---------------")
							askResponse = await impl.run(request_json["question"], request_json.get("overrides") or {})
						# Extract and print the "token_usage" value
						token_usage = askResponse.get("token_usage")
						askResponse['dbresponse'] = 0
					else:
						askResponse['answer'] = match_found[0]['answer']
						askResponse['thoughts'] = match_found[0]['thoughts']
						askResponse['data_points'] = ast.literal_eval(match_found[0]['data_points'])
						askResponse['dbresponse'] = 1
						askResponse['token_usage'] = 0
						askResponse['credit_used'] = 0
				else:
						askResponse['answer']  = qa_record['answer']
						askResponse['dbresponse'] = 1
						askResponse['thoughts'] = qa_record['thoughts']
						askResponse['data_points'] = qa_record['data_points']
						askResponse['token_usage'] = qa_record['token_usage']
						askResponse['credit_used'] = qa_record["credit_used"]

			else:
				impl = current_app.config[CONFIG_ASK_APPROACHES].get(approach)
				print("-----Ask Step 1---------------")
				if not impl:
					return jsonify({"error": "unknown approach"}), 400
				# Workaround for: https://github.com/openai/openai-python/issues/371
				async with aiohttp.ClientSession() as s:
					openai.aiosession.set(s)
					print("-----Ask Step 2---------------")
					askResponse = await impl.run(request_json["question"], request_json.get("overrides") or {})
				# Extract and print the "token_usage" value
				token_usage = askResponse.get("token_usage")
				askResponse['dbresponse'] = 0
			credit_used = credit_used_by_query(token_usage)
			balance=round(calculate_balance(email)-credit_used, 2)
			update_transactions_table(email=email, balance=balance, service_type = "Query", token_usage= token_usage, credit_used= credit_used)
			qa_record = insert_qa_records(query, askResponse['answer'], start_time, askResponse['thoughts'], askResponse['data_points'], exclude_category, token_usage, email, credit_used)
			askResponse['uid'] = str(qa_record['id'])
			askResponse['balance'] = balance
			return askResponse
		except Exception as e:
			print(str(e))
			logging.exception("Exception in /ask")
		return jsonify({"error": str(e)}), 500
	else:
		print("Insufficient balance for user:", email)
		return jsonify({"Message": "Insufficient balance. Please add credits to proceed further"}), 403

@bp.route("/chat", methods=["POST"])
async def chat():
	print("-----------------Chat API Call happened-------------")
	data = await request.get_json()
	email = data.get("email")
	if check_balance(email):
		if not request.is_json:
			return jsonify({"error": "request must be json"}), 415
		request_json = await request.get_json()
		approach = request_json["approach"]
		query = request_json["history"][0].get("user")
		exclude_category = request_json.get("overrides", {}).get("exclude_category")
		# Call get_cat_status to fetch categories with status 0
		cat_list = await get_inactive_categories()
		exclude_category.extend(cat['id'] for cat in cat_list)
		request_json["overrides"]["exclude_category"] = exclude_category
		try:
			start_time = time.time()
			impl = current_app.config[CONFIG_CHAT_APPROACHES].get(approach) 
			print("-----Chat Step 1---------------")
			if not impl:
				return jsonify({"error": "unknown approach"}), 400
			# Workaround for: https://github.com/openai/openai-python/issues/371
			async with aiohttp.ClientSession() as s:
				openai.aiosession.set(s)
				print("-----Chat Step 1---------------")
				r = await impl.run(request_json["history"], request_json.get("overrides") or {})
			# Extract and print the "token_usage" value
			token_usage = r.get("token_usage")
			credit_used = credit_used_by_query(token_usage)
			balance=round(calculate_balance(email)-credit_used, 2)
			update_transactions_table(email=email, balance=balance, service_type = "Query", token_usage= token_usage, credit_used= credit_used)
			insert_qa_records(query, r['answer'], start_time, r['thoughts'], r['data_points'], exclude_category, token_usage, email, credit_used)
			r['balance'] = balance
			return jsonify(r)
		except Exception as e:
			logging.exception("Exception in /chat")
			return jsonify({"error": str(e)}), 500
	else:
		print("Insufficient balance for user:", email)
		return jsonify({"Message": "Insufficient balance. Please add credits to proceed further"}), 403	

#@bp.before_request
#async def ensure_openai_token():
#    openai_token = current_app.config[CONFIG_OPENAI_TOKEN]
#    if openai_token.expires_on < time.time() + 60:
#        openai_token = await current_app.config[CONFIG_CREDENTIAL].get_token("https://cognitiveservices.azure.com/.default")
#        #current_app.config[CONFIG_OPENAI_TOKEN] = openai_token
#        openai.api_key = openai_token.token

@bp.before_app_serving
async def setup_clients():

	# Replace these with your own values, either in environment variables or directly here
	AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
	AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME")
	AZURE_QNA_INDEX = os.getenv("AZURE_QA_INDEX_NAME")
	AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

	AZURE_STORAGE_ACCOUNT = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
	AZURE_STORAGE_CONTAINER = os.getenv("BLOB_STORAGE_CONTAINER_NAME")

	AZURE_OPENAI_SERVICE_BASE = os.getenv("AZURE_OPENAI_API_BASE")
	AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
	AZURE_OPENAI_TYPE = os.getenv("OPENAI_API_TYPE")
	AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("GPT3_LLM_MODEL_DEPLOYMENT_NAME")
	AZURE_OPENAI_CHATGPT_MODEL = os.getenv("GPT3_LLM_MODEL_NAME")
	AZURE_OPENAI_EMB_DEPLOYMENT = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME_1")
	AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")

	KB_FIELDS_CONTENT = 'content'
	KB_FIELDS_SOURCEPAGE = 'sourcepage'

	COSMOS_DATABASE_ID = os.getenv("COSMOS_DB_NAME")
	COSMOS_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT_URL")
	COSMOS_KEY = os.getenv("COSMOS_DB_PRIMARY_KEY")
	COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")


	print("--------------Starting Backend Setup-------------")
	#--------------------Creating INDEXES in ACS--------------------
	key = AZURE_SEARCH_ADMIN_KEY
	azure_search_credential = AzureKeyCredential(key)

	vector_search = VectorSearch(
		algorithm_configurations=[
			VectorSearchAlgorithmConfiguration(
				name="my-vector-config",
				kind="hnsw",
				hnsw_parameters={
					"m": 4,
					"efConstruction": 400,
					"efSearch": 1000,
					"metric": "cosine"
				}
			)
		]
	)


	#-------- STEP 1 SEARCH INDEX-----------
	index_client = SearchIndexClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, credential=azure_search_credential)
	fields = [
		SimpleField(name="id", type=SearchFieldDataType.String, key=True),
		SearchableField(name="title", type=SearchFieldDataType.String,searchable=True, retrievable=True,filterable=True),
		SearchableField(name="content", type=SearchFieldDataType.String,searchable=True, retrievable=True),
		SearchableField(name="category", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="sourcepage", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=1536, vector_search_configuration="my-vector-config"),
	]

	semantic_config = SemanticConfiguration(
		name="my-semantic-config",
		prioritized_fields=PrioritizedFields(
			title_field=SemanticField(field_name="title"),
			prioritized_content_fields=[SemanticField(field_name="content")]
		)
	)

	semantic_settings = SemanticSettings(configurations=[semantic_config])

	# Create the search index with the semantic settings
	index = SearchIndex(name=AZURE_SEARCH_INDEX, fields=fields,vector_search=vector_search, semantic_settings=semantic_settings)
	index_client.create_or_update_index(index)
	print('---------- Search Index Created------------------')

	#-------- QNA INDEX-----------
	qa_index_client = SearchIndexClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, credential=azure_search_credential)

	qa_fields = [
		SimpleField(name="id", type=SearchFieldDataType.String, key=True),
		SearchableField(name="question", type=SearchFieldDataType.String,searchable=True, retrievable=True),
		SearchableField(name="answer", type=SearchFieldDataType.String,searchable=True, retrievable=True),
		SearchableField(name="exclude_category", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="index_format", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="tracing", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="cost_saved", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="data_points", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="thoughts", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=1536, vector_search_configuration="my-vector-config"),
	]

	qa_semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=PrioritizedFields(
        title_field=SemanticField(field_name="answer"),
        prioritized_content_fields=[SemanticField(field_name="question")]
    )
)

	# Create the semantic settings with the configuration
	qa_semantic_settings = SemanticSettings(configurations=[qa_semantic_config])

	qa_index = SearchIndex(name=AZURE_QNA_INDEX, fields=qa_fields,vector_search=vector_search, semantic_settings=qa_semantic_settings)
	qa_index_client.create_or_update_index(qa_index)
	print('---------- QNA Index Created------------------')

	#-------- STEP 2 CREATE ACS SEARCH CLIENT-----------
	# Set up clients for Cognitive Search and Storage
	search_client = SearchClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,index_name=AZURE_SEARCH_INDEX, credential=azure_search_credential)
	qna_client = SearchClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,index_name=AZURE_QNA_INDEX, credential=azure_search_credential)
	print('---------- ACS SEARCH CLIENT Created------------------')


	#-------- STEP 3 BLOB STORAGE SETTING-----------
	blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_ACCOUNT)
	blob_container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)

	current_app.config['blob_container_client'] = blob_container_client
	print('---------- Blob Storage Setup Done------------------')

	#-------- STEP 4 AZURE OPEN AI SETTING-----------
	openai.api_base = AZURE_OPENAI_SERVICE_BASE
	openai.api_version = AZURE_OPENAI_VERSION
	openai.api_type = AZURE_OPENAI_TYPE
	openai.api_key = AZURE_OPENAI_KEY
	print('---------- Azure OpenAI Setup Done------------------')

	#-------- STEP 5 SETTING UP COSMOS DB-----------

	Cosmos_db_client = CosmosClient(url=COSMOS_ENDPOINT, credential=COSMOS_KEY)
	database = Cosmos_db_client.get_database_client(COSMOS_DB_NAME)
	current_app.config['cosmos_db'] = database

	#-------- STEP 6 SETTING UP CURRENT CONFIG-----------
	# Various approaches to integrate GPT and external knowledge, most applications will use a single one of these patterns
	# or some derivative, here we include several for exploration purposes

	current_app.config[CONFIG_ASK_APPROACHES] = {
		"rtr": RetrieveThenReadApproach(
			search_client,
			AZURE_OPENAI_CHATGPT_DEPLOYMENT,
			AZURE_OPENAI_CHATGPT_MODEL,
			AZURE_OPENAI_EMB_DEPLOYMENT,
			KB_FIELDS_SOURCEPAGE,
			KB_FIELDS_CONTENT
		),
		"rrr": ReadRetrieveReadApproach(
			search_client,
			AZURE_OPENAI_CHATGPT_DEPLOYMENT,
			AZURE_OPENAI_EMB_DEPLOYMENT,
			KB_FIELDS_SOURCEPAGE,
			KB_FIELDS_CONTENT
		),
		"rda": ReadDecomposeAsk(search_client,
			AZURE_OPENAI_CHATGPT_DEPLOYMENT,
			AZURE_OPENAI_EMB_DEPLOYMENT,
			KB_FIELDS_SOURCEPAGE,
			KB_FIELDS_CONTENT
		)
	}

	current_app.config[CONFIG_CHAT_APPROACHES] = {
		"rrr": ChatReadRetrieveReadApproach(
			search_client,
			AZURE_OPENAI_CHATGPT_DEPLOYMENT,
			AZURE_OPENAI_CHATGPT_MODEL,
			AZURE_OPENAI_EMB_DEPLOYMENT,
			KB_FIELDS_SOURCEPAGE,
			KB_FIELDS_CONTENT,
		)
	}
	print("-------------------SETPUP COMPLETED------------------")


	# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed,
	# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the
	# keys for each service
	# If you encounter a blocking error during a DefaultAzureCredential resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
	azure_credential = DefaultAzureCredential(exclude_shared_token_cache_credential = True)
	current_app.config[CONFIG_CREDENTIAL] = azure_credential
	current_app.config[CONFIG_BLOB_CONTAINER_CLIENT] = blob_container_client
