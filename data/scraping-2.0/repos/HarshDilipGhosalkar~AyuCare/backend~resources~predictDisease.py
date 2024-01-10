import json
import os
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

model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
service_context = ServiceContext.from_defaults(embed_model=model)
set_global_service_context(service_context)

storage_context = StorageContext.from_defaults(persist_dir="./AI/medical_index")
medical_index = load_index_from_storage(storage_context)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

predictDiseaseChatEngine = medical_index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="""I will tell you the symptoms I am having. You should analyse them and tell me what disease I may be having and The ayurvedic treatment for that disease. 
    Format the symptoms as follows. Output should be strictly in json format and should strictly contain nothing extra before and after that:
    {"DiseaseName": "XYZ",
    "Treatment": [
        {
            "treatmentName": "XYZ",
            "treatment": "XYZ"
        }
    ]}
    Symptoms are:""" ,
)

class PredictDisease(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("symptoms", type=str, required=True, help="symptoms is required")
        args = parser.parse_args()
        symptoms = args["symptoms"]

        try:
            response = predictDiseaseChatEngine.chat(symptoms)
            print(response)
            json_response = json.loads(response.response)
            print(json_response)
            json_response["symptoms"] = symptoms
            return {"error": False, "data": json_response}, 200
        except Exception as e:
            return {"error": True, "message": str(e)}, 500
    