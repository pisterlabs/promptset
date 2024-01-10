from os import getenv
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Body, FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse as FastAPIJSONResponse
from langchain.llms.base import LLM

from .chain import embed_sources, process_query
from .ocs_utils import AppAPIAuthMiddleware, get_nc_url
from .utils import value_of
from .vectordb import BaseVectorDB

load_dotenv()

app = FastAPI(debug=getenv('DEBUG', '0') == '1')


# middlewares

app.add_middleware(
	CORSMiddleware,
	allow_origins=[get_nc_url()],
	allow_methods=['*'],
	allow_headers=['*'],
)

if value_of(getenv('DISABLE_AAA', '0')) == '0':
	app.add_middleware(AppAPIAuthMiddleware)


def JSONResponse(
	content: Any = 'ok',
	status_code: int = 200,
	**kwargs
) -> FastAPIJSONResponse:
	'''
	Wrapper for FastAPI JSONResponse
	'''
	if isinstance(content, str):
		if status_code >= 400:
			return FastAPIJSONResponse(
				content={ 'error': content },
				status_code=status_code,
				**kwargs,
			)
		return FastAPIJSONResponse(
			content={ 'message': content },
			status_code=status_code,
			**kwargs,
		)

	return FastAPIJSONResponse(content, status_code, **kwargs)


@app.get('/')
def _(request: Request):
	'''
	Server check
	'''
	return f'Hello, {request.headers.get("username", "anon")}!'


# TODO: for testing, remove later
@app.get('/world')
def _(query: str | None = None):
	em = app.extra.get('EMBEDDING_MODEL')
	return em.embed_query(query if query is not None else 'what is an apple?')


# TODO: for testing, remove later
@app.get('/vectors')
def _(userId: str):
	from chromadb import ClientAPI
	from .utils import COLLECTION_NAME

	db: BaseVectorDB = app.extra.get('VECTOR_DB')
	client: ClientAPI = db.client
	db.setup_schema(userId)

	return JSONResponse(
		client.get_collection(COLLECTION_NAME(userId)).get()
	)


@app.put('/enabled')
def _(enabled: bool):
	print(f'{enabled:}')
	return JSONResponse(content={'error': ''}, status_code=200)


@app.get('/heartbeat')
def _():
	print('heartbeat_handler: result=ok')
	return JSONResponse(content={'status': 'ok'}, status_code=200)


@app.post('/deleteSources')
def _(userId: Annotated[str, Body()], sourceNames: Annotated[list[str], Body()]):
	sourceNames = [source.strip() for source in sourceNames if source.strip() != '']

	if len(sourceNames) == 0:
		return JSONResponse('No sources provided', 400)

	db: BaseVectorDB = app.extra.get('VECTOR_DB')

	if db is None:
		return JSONResponse('Error: VectorDB not initialised', 500)

	source_objs = db.get_objects_from_sources(userId, sourceNames)
	res = db.delete_by_ids(userId, [
		source.get('id')
		for source in source_objs.values()
		if value_of(source.get('id') is not None)
	])

	# NOTE: None returned in `delete_by_ids` should have meant an error but it didn't in the case of
	# weaviate maybe because of the way weaviate wrapper is implemented (langchain's api does not take
	# class name as input, which will be required in future versions of weaviate)
	if res is None:
		print('Deletion query returned "None". This can happen in Weaviate even if the deletion was \
successful, therefore not considered an error for now.')

	if res is False:
		return JSONResponse('Error: VectorDB delete failed, check vectordb logs for more info.', 400)

	return JSONResponse('All valid sources deleted')


@app.put('/loadSources')
def _(sources: list[UploadFile]):
	if len(sources) == 0:
		return JSONResponse('No sources provided', 400)

	# TODO: headers validation using pydantic
	if not all([
		value_of(source.headers.get('userId'))
		and value_of(source.headers.get('type'))
		and value_of(source.headers.get('modified'))
		for source in sources]
	):
		return JSONResponse('Invaild/missing headers', 400)

	db: BaseVectorDB = app.extra.get('VECTOR_DB')
	if db is None:
		return JSONResponse('Error: VectorDB not initialised', 500)

	result = embed_sources(db, sources)
	if not result:
		return JSONResponse('Error: All sources were not loaded, check logs for more info', 500)

	return JSONResponse('All sources loaded')


@app.get('/query')
def _(userId: str, query: str, useContext: bool = True, ctxLimit: int = 5):
	llm: LLM = app.extra.get('LLM_MODEL')
	if llm is None:
		return JSONResponse('Error: LLM not initialised', 500)

	db: BaseVectorDB = app.extra.get('VECTOR_DB')
	if db is None:
		return JSONResponse('Error: VectorDB not initialised', 500)

	template = app.extra.get('LLM_TEMPLATE')

	output, sources = process_query(
		user_id=userId,
		vectordb=db,
		llm=llm,
		query=query,
		use_context=useContext,
		ctx_limit=ctxLimit,
		**({'template': template} if template else {}),
	)

	if output is None:
		return JSONResponse('Error: check if the model specified supports the query type', 500)

	return JSONResponse({
		'output': output,
		'sources': sources,
	})
