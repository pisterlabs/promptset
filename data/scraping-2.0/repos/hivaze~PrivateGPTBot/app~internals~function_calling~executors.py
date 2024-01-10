import json
import logging
from typing import List

from langchain.vectorstores.chroma import Chroma

from app import settings
from app.database.sql_db_service import UserEntity, TokensPackageEntity
from app.internals.chat.chat_history import FunctionCallMessage, FunctionResponseMessage

from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

from app.utils.misc import clean_text

logger = logging.getLogger(__name__)
html2text = Html2TextTransformer()


def website_request(user: UserEntity, current_user_data: dict, url: str):
    """Parses information from a website via a link 'url'"""
    try:
        # vector_store: Chroma = current_user_data.get('vectorstore')
        loader = AsyncHtmlLoader([url])
        docs = loader.load()
        docs = html2text.transform_documents(docs)
        docs = [clean_text(doc.page_content) for doc in docs]
        total_symbols = sum([len(doc) for doc in docs])
        if total_symbols > 50_000:
            return ("ERROR: The content of this website is too long for the assistant."
                    " The assistant only works with short and medium-length pages (for example, news pages)")
    except Exception as e:
        logger.info(f"Exception while getting html from {url}: {e} for user '{user.user_id}' | '{user.user_name}'")
        return ("ERROR: Most likely, the site is technically inaccessible to the assistant. "
                "This may be due to the fact that it is protected from scanning by bots.")
    return {"web_content": docs}


def search_in_document_query(user: UserEntity, current_user_data: dict, document_id: str, query: str):
    """Executes information search by 'query' in the document with id 'document_id'"""
    vector_store: Chroma = current_user_data.get('vectorstore')
    found_documents = vector_store.search(query,
                                          search_type='similarity',
                                          k=settings.config.documents.search_best_k,
                                          filter={'file_id': document_id})
    logger.info(f"Found results: {found_documents}")
    found_documents = [doc.page_content for doc in found_documents]
    return {"found": found_documents}


def get_tokens_balance(user: UserEntity, current_user_data: dict):
    tokens_packages: List[TokensPackageEntity] = user.tokens_packages
    available_tokens = [tokens_package.left_tokens for tokens_package in tokens_packages]
    package_names = [tokens_package.package_name for tokens_package in tokens_packages]
    expires_at = [tokens_package.expires_at for tokens_package in tokens_packages]
    return [
        {
            'package_name': j,
            'left_tokens': i,
            'expires_at': k.strftime("%Y-%m-%d %H:%M")
        } for i, j, k in zip(available_tokens, package_names, expires_at)
    ]


FUNCTIONS_MAPPING: dict = {
    'website_request': website_request,
    'search_in_document_query': search_in_document_query,
    'get_tokens_balance': get_tokens_balance
}


def execute_function_call(user: UserEntity,
                          current_user_data: dict,
                          message: FunctionCallMessage) -> FunctionResponseMessage:
    func_name = message.name
    assert func_name in FUNCTIONS_MAPPING.keys(), f"Function '{func_name}' is not supported"

    logger.info(f"Calling function '{func_name}' with args '{message.arguments}' for user '{user.user_id}'")
    results = FUNCTIONS_MAPPING[func_name](user, current_user_data, **message.arguments)

    return FunctionResponseMessage(name=func_name, text=json.dumps(results, ensure_ascii=False))
