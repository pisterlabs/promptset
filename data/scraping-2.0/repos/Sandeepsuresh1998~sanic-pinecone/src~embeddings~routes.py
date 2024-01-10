from sanic import Blueprint
from sanic.response import json
from utils.constants import APP_NAME
from sanic import Sanic
from sanic_ext import validate
from embeddings.types import EmbeddingCreate
from llama_index.embeddings import OpenAIEmbedding
from llama_index import VectorStoreIndex
from openai import OpenAI
import pinecone
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

bp = Blueprint("embeddings")


@bp.post("/v1/embeddings/create")
@validate(json=EmbeddingCreate)
async def create(request, body: EmbeddingCreate):

   ## TODO: There will need to be an auth decorator incorporated with Auth0 here

   # Create the embeddings for the loaded text
   text_input = body.text
   sanic_app = Sanic.get_app(APP_NAME)

   # Generate the embedding with OpenAI
   openai = OpenAI(
      api_key=sanic_app.config.SANIC_OPENAI_KEY
   )

   ## TODO: Error checking and retries built in case of failure
   embeddings_vector = openai.embeddings.create(
      input=[text_input], 
      model="text-embedding-ada-002",
   ).data[0].embedding

   # Create connection to pinecone
   openapi_config = OpenApiConfiguration.get_default_copy()
   openapi_config.verify_ssl = False # Wasn't working locally
   ## TODO: This should really be long living in the sanic app context
   pinecone.init(
      api_key=sanic_app.config.SANIC_PINECONE_KEY,
      environment="gcp-starter",
      openapi_config=openapi_config,
    )

   # Retrieve the index
   journal_index = pinecone.Index("journal")
   journal_index.upsert(
      vectors=[
         {
            'id': body.user_id, 
            "values":embeddings_vector, 
            "metadata": {
               'user_id': body.user_id,
               'text': text_input  
            }
         },
      ],
    )
   
   return json({
      "success": "true",
   })

@bp.post("v1/embeddings/clear")
async def clear(request):
   ## TODO: This is really only for testing purposes and should be removed in production
   sanic_app = Sanic.get_app(APP_NAME)

   # Create connection to pinecone
   openapi_config = OpenApiConfiguration.get_default_copy()
   openapi_config.verify_ssl = False


   pinecone.init(
      api_key=sanic_app.config.SANIC_PINECONE_KEY,
      environment="gcp-starter",
      openapi_config=openapi_config,
   )
   
   # Retrieve the index
   journal_index = pinecone.Index("journal")

   journal_index.delete(
      ids=["kljsndfkjn-skdjbnfs"],
   )

   return json(
      {"success": "true"}
   )