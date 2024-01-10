from flask_restx import Resource, Namespace
from flask import request, current_app as app
from app.main.utils.auth import token_required
from app.main.utils.logger import addEmailForLogger
from app.main.utils.dependencyInjection import DependencyInjection
import openai
api = Namespace("api")

@api.route("/summarizeContent")
class summarizeContentResource(Resource):
    @token_required
    def post(self, *args, **kwargs):
        openai.api_key = kwargs.get("api_key") or "invalid"
        email = kwargs.get("email")
        json = request.get_json()
        result = DependencyInjection().getTextOperationService(app).getSummarizeContent(json.get("content"))
        return addEmailForLogger(result, email)
@api.route("/fixTypos")
class fixTyposResource(Resource):
    @token_required
    def post(self, *args, **kwargs):
        openai.api_key = kwargs.get("api_key") or "invalid"
        email = kwargs.get("email")
        json = request.get_json()
        result = DependencyInjection().getTextOperationService(app).getFixTypos(json.get("content"))
        return addEmailForLogger(result, email)

@api.route("/explainCode")
class explainCodeResource(Resource):
    @token_required
    def post(self, *args, **kwargs):
        openai.api_key = kwargs.get("api_key") or "invalid"
        email = kwargs.get("email")
        languagePreference = kwargs.get("language_preference")
        json = request.get_json()
        result = DependencyInjection().getTextOperationService(app).getExplainCode(json.get("content"), languagePreference)
        return addEmailForLogger(result, email)