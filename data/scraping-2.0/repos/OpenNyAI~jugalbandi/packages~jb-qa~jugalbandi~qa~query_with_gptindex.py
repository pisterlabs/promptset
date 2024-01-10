import openai
import json
from llama_index import load_index_from_storage, StorageContext
from jugalbandi.core.errors import InternalServerException, ServiceUnavailableException
from jugalbandi.document_collection import DocumentCollection


async def querying_with_gptindex(document_collection: DocumentCollection, query: str):
    index_content = await document_collection.read_index_file("gpt-index", "index.json")
    index_content = index_content.decode('utf-8')
    index_dict = json.loads(index_content)
    storage_context = StorageContext.from_dict(index_dict)
    index = load_index_from_storage(storage_context=storage_context)
    query_engine = index.as_query_engine()
    try:
        response = query_engine.query(query)
        source_nodes = response.source_nodes
        source_text = []
        for i in range(len(source_nodes)):
            text = source_nodes[i].node.get_text().strip()
            source_text.append(text)
        return str(response).strip(), source_text
    except openai.error.RateLimitError as e:
        raise ServiceUnavailableException(
            f"OpenAI API request exceeded rate limit: {e}"
        )
    except (openai.error.APIError, openai.error.ServiceUnavailableError):
        raise ServiceUnavailableException(
            "Server is overloaded or unable to answer your request at the moment."
            " Please try again later"
        )
    except Exception as e:
        raise InternalServerException(e.__str__())
