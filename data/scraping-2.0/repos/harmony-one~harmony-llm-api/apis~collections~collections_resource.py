import shutil
from flask import request, jsonify, make_response, current_app
from flask_restx import Namespace, Resource
from openai import OpenAIError
import json
import threading
from llama_index.llms.base import ChatMessage
from res import EngMsg as msg
from storages import chromadb
from res import PdfFileInvalidFormat, InvalidCollectionName, InvalidCollection
from .collections_helper import CollectionHelper
from models import db
from services import WebCrawling, PdfHandler
from models import CollectionError

api = Namespace('collections', description=msg.API_NAMESPACE_LLMS_DESCRIPTION)

def data_generator(response):
    for chunk in response:
        yield f"data: {json.dumps(chunk)}\n\n"

collection_helper = CollectionHelper(chromadb)

@api.route('/reset', doc=False)
class CollectionHandler(Resource):
    def post(self):
        """
        Endpoint that resets Chromadb
        """
        current_app.logger.info('Reseting Chromadb')
        try:
            if (collection_helper.reset_database()):
                # collection_helper.delete_folders()
                return 'OK', 204
            else:
                return make_response(jsonify({"error": "An unexpected error occurred."}), 500)
        except Exception as e:
            current_app.logger.error(e)
            error_message = str(e)
            current_app.logger.error(f"Unexpected Error: {error_message}")
            return make_response(jsonify({"error": "An unexpected error occurred."}), 500)

@api.route('/document')
class AddDocument(Resource):
    def post(self):
        """
        Endpoint that creates collections
        Receives any document (URL, PDF, voice) and creates a collection (Vector index).
        Returns a collection ID
        """
        current_app.logger.info('handling document')
        data = request.json
        chat_id = data.get('chatId')
        url = data.get('url')
        file_name = data.get('fileName')
        try:
            if (chat_id and url):
                collection_name = collection_helper.get_collection_name(chat_id, url)
                thread = threading.Thread(target=self.__collection_request_handler, args=(url, collection_name, file_name, current_app.app_context()))
                thread.start()
                return make_response(jsonify({"collectionName": f"{collection_name}"}), 200)
            else:
                return make_response(jsonify({"error": "Bad request, parameters missing"}), 400)
        except Exception as e:
            error_message = str(e)
            current_app.logger.error(f"Unexpected Error: {error_message}")
            return make_response(jsonify({"error": "An unexpected error occurred."}), 500)

    def __collection_request_handler(self, url, collection_name, file_name, context):
        try:
            if (not collection_helper.is_pdf_url(url)):
                crawl = WebCrawling()
                text_array = crawl.get_web_content(url)
                if text_array.get('urlText').__len__() > 0:
                    collection_helper.db.store_text_array_from_url(text_array, collection_name)
                else:
                    raise InvalidCollection('Invalid collection')
            else:
                pdf_handler = PdfHandler()
                chunks = pdf_handler.pdf_to_chunks(url)
                if (chunks.__len__() > 0):
                    collection_helper.db.store_text_array(chunks, collection_name)
                else:
                    raise InvalidCollection('Invalid collection')
        except (Exception, InvalidCollection) as e:
            context.push()  
            error = CollectionError(dict( collection_name = collection_name))
            error.save()

@api.route('/document/<collection_name>')
class CheckDocument(Resource):

    @api.doc(params={"collection_name": msg.API_DOC_PARAMS_COLLECTION_NAME})
    def get(self, collection_name):
        """
        Endpoint that checks collection creation status.
        If collection exists, returns indexing price
        """
        try:
            current_app.logger.info('Checking collection status')
            if (collection_name): 
                collection_error = CollectionError.query.filter_by(collection_name=collection_name).first()
                if (collection_error):
                    response = {
                        "price": -1, # TBD
                        "status": 'DONE',
                        "error": 'INVALID_COLLECTION'
                    }
                else:
                    collection = collection_helper.get_collection(collection_name)
                    if (collection):
                        embeddings_number = collection.count()
                        current_app.logger.info(f'******* Number of embeddings: {embeddings_number}')
                        response = {
                            "price": embeddings_number * 0.05, # TBD
                            "status": 'DONE',
                            "error": None
                        }
                    else:
                        response = {
                            "price": 0,
                            "status": 'PROCESSING',
                            "error": None
                        }
                return make_response(jsonify(response), 200)
            else:
                return "Bad request, parameters missing", 400
        except Exception as e:
            current_app.logger.error(e)
            error_message = str(e)
            current_app.logger.error(f"Unexpected Error: {error_message}")
            return make_response(jsonify({"error": "An unexpected error occurred."}), 500)
    
    @api.doc(params={"collection_name": msg.API_DOC_PARAMS_COLLECTION_NAME})
    def delete(self, collection_name):
        """
        Endpoint that deletes a collection
        """            
        try:
            current_app.logger.info('Deleting collection')
            if (collection_name):
                # collection_helper.delete_collection(collection_name)
                return 'OK', 204
            else:
                return "Bad request, parameters missing", 400
        except ValueError as e:
            return 'OK', 204
        except Exception as e:
            error_message = str(e)
            current_app.logger.error(f"Unexpected Error: {error_message}")
            return make_response(jsonify({"error": "An unexpected error occurred."}), 500)

@api.route('/query')
class WebCrawlerTextRes(Resource):
    # 
    # @copy_current_request_context
    def post(self):
        """
        Endpoint to handle LLMs request.
        Receives a message from the user, processes it, and returns a response from the model.
        """ 
        data = request.json
        prompt = data.get('prompt')
        collection_name = data.get('collectionName')
        url = data.get('url')
        conversation = data.get('conversation')
        chat_history = [ChatMessage(content=item.get('content'), role=item.get('role')) for item in conversation]
        try:
            current_app.logger.info(f'Inquiring a collection {collection_name}')
            if collection_name:
                response = collection_helper.collection_query(collection_name, prompt, chat_history) 
                return make_response(jsonify(response), 200)
            else:
                current_app.logger.error('Bad request')
                return make_response(jsonify({"error": "Bad request"}), 400)
        except InvalidCollectionName as e:
            
            current_app.logger.error(e)
            return make_response(jsonify({"error": e.args[1]}), 404)   
        except OpenAIError as e:
            # Handle OpenAI API errors
            error_message = str(e)
            current_app.logger.error(f"OpenAI API Error: {error_message}")
            return jsonify({"error": error_message}), 500
        except Exception as e:
            error_message = str(e)
            current_app.logger.error(f'ERROR ***************: {error_message}')
            current_app.logger.error(e)
            return make_response(jsonify({"error": "An unexpected error occurred."}), 500)


@api.errorhandler(PdfFileInvalidFormat)
def PdfHandlingError():
    current_app.logger.error(f"Unexpected Error: {'PDF file not supported/readable'}")
    return 'PDF file not supported/readable', 415