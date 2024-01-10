from .utils import FacilityABC
from Agently.utils import RuntimeCtx, RuntimeCtxNamespace

class Embedding(FacilityABC):
    def __init__(self, *, storage: object, plugin_manager: object, settings: object):
        self.plugin_manager = plugin_manager
        self.settings = settings

    def OpenAI(self, text: str, options: dict={}):
        from openai import OpenAI
        import httpx
        settings = RuntimeCtxNamespace("embedding.OpenAI", self.settings)
        # Prepare Client Params
        client_params = {}
        base_url = settings.get_trace_back("url")
        if base_url:
            client_params.update({ "base_url": base_url })
        proxy = self.settings.get_trace_back("proxy")
        if proxy:
            client_params.update({ "http_client": httpx.Client( proxies = proxy ) })
        api_key = settings.get_trace_back("auth.api_key")
        if api_key:
            client_params.update({ "api_key": api_key })
        else:
            raise Exception("[Embedding] OpenAI require api_key. use Agently.facility.set_settings('embedding.OpenAI.auth', { 'api_key': '<Your-API-Key>' }) to set it.")
        if "model" not in options:
            options["model"] = "text-embedding-ada-002"
        # Create Client
        client = OpenAI(**client_params)
        # Request
        result = client.embeddings.create(
            input = text.replace("\n", " "),
            **options
        )
        if result.data:
            return result.data[0].embedding
        else:
            raise Exception(f"[Embedding] OpenAI Error: { dict(result) }")

    def ERNIE(self, text_list: (str, list), options: dict={}):
        import erniebot
        settings = RuntimeCtxNamespace("embedding.ERNIE", self.settings)
        erniebot.api_type = "aistudio"
        access_token = settings.get_trace_back("auth.aistudio")
        if access_token:
            erniebot.access_token = access_token
        else:
            raise Exception("[Embedding] ERNIE require aistudio access token. use Agently.facility.set_settings('embedding.ERNIE.auth', { 'aistudio': '<Your-Access-Token>' }) to set it.")
        if "model" not in options:
            options["model"] = "ernie-text-embedding"
        if not isinstance(text_list, list):
            text_list = [text_list]
        result = erniebot.Embedding.create(
            input = text_list,
            **options
        )
        result = dict(result)
        if result["rcode"] == 200:
            return result["rbody"]["data"]
        else:
            raise Exception(f"[Embedding] ERNIE Error: { result }")

    def ZhipuAI(self, text: str, options: dict={}):
        import zhipuai
        settings = RuntimeCtxNamespace("embedding.ZhipuAI", self.settings)
        api_key = settings.get_trace_back("auth.api_key")
        if api_key:
            zhipuai.api_key = api_key
        else:
            raise Exception("[Embedding] ZhipuAI require api_key. use Agently.facility.set_settings('embedding.ZhipuAI.auth', { 'api_key': '<Your-API-Key>' }) to set it.")
        if "model" not in options:
            options["model"] = "text_embedding"
        result = zhipuai.model_api.invoke(
            prompt = text,
            **options
        )
        if "data" in result:
            return result["data"]["embedding"]
        else:
            raise Exception(f"[Embedding] ZhipuAI Error: { result }")

    def Google(self, text: str, options: dict={}):
        import json
        import httpx
        settings = RuntimeCtxNamespace("embedding.Google", self.settings)
        api_key = settings.get_trace_back("auth.api_key")
        if api_key == None:
            raise Exception("[Embedding] Google require api_key. use Agently.facility.set_settings('embedding.google.auth', { 'api_key': '<Your-API-Key>' }) to set it.")
        proxy = self.settings.get_trace_back("proxy")
        model = "embedding-001"
        if "model" in options:
            model = options["model"]
            del options["model"]
        request_params = {
            "data": json.dumps({
                "content": {
                    "parts": [{ "text": text }]
                },
                **options
            }),
            "timeout": None,
        }
        result = httpx.post(
            f"https://generativelanguage.googleapis.com/v1/models/{ model }:embedContent?key={ api_key }",
            **request_params
        )
        if result.status_code:
            return result.json()["embedding"]["values"]
        else:
            raise Exception(f"[Embedding] Google Error: { dict(result) }")

def export():
    return ("embedding", Embedding)