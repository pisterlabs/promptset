from utils.constants import APP_NAME
from sanic import Sanic
from sanic import Blueprint
from sanic_ext import validate
from embeddings.types import EmbeddingCreate
from llama_index.embeddings import OpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from sanic import json

bp = Blueprint("embeddings")

@bp.post("/v1/embeddings/create")
@validate(json=EmbeddingCreate)
async def create(request, body: EmbeddingCreate):

   embeddings_model = OpenAIEmbedding()
   app = Sanic.get_app(APP_NAME)
   journal_index = app.ctx.journal_index
   journal_index.upsert(
    vectors=[
       {'id': "vec1", "values":[0.1, 0.2, 0.3, 0.4], "metadata": {'genre': 'drama'}},
       {'id': "vec2", "values":[0.2, 0.3, 0.4, 0.5], "metadata": {'genre': 'action'}},
    ],
   )
   return json({"hello": "world"})