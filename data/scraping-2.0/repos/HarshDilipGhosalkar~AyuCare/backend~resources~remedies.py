import os
import pymongo
from flask_restful import Resource, reqparse
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    LangchainEmbedding,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from llama_index.memory import ChatMemoryBuffer

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OPENAI_API_KEY"] = os.getenv("FLASK_OPENAI_API_KEY")

try:
    client = pymongo.MongoClient(os.getenv("FLASK_MONGODB_URI"))
    db = client["hackathon"]
    print("Connected to the dev database successfully.")
except pymongo.errors.ConnectionFailure as e:
    print("Failed to connect to the dev database: %s", e)

medicineRef = db["medicine"]

model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
service_context = ServiceContext.from_defaults(embed_model=model)
set_global_service_context(service_context)

storage_context = StorageContext.from_defaults(persist_dir="./AI/disease_remedy_index")
medical_index = load_index_from_storage(storage_context)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

ayurvedic_remedy_engine = medical_index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="""
        Act as an ayurvedic doctor suggesting ayurvedic remedies for disease. 
        I will provide you disease name, you should provide me ayurvedic remedies for that disease in brief.
    """ ,
)

class AyurvedicRemedies(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("disease", type=str, required=True, help="disease is required")

        args = parser.parse_args()
        disease = args["disease"]
        data = {
            "disease": disease,
            "remedy": getAyurvedicRemedies(disease)
        }

        return {"error": False, "data": data}, 200


def getAyurvedicRemedies(disease):
    try:
        response = ayurvedic_remedy_engine.chat(disease)
        return response.response
    except Exception as e:
        return {"error": True, "message": str(e)}, 500
