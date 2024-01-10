from langchain import HuggingFaceTextGenInference


class LLMFactory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Ensure that initialization code is only run once
        if self._initialized:
            return
        inference_server_url_local = "http://127.0.0.1:8080"
        self.fqlLLM = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url_local,
            temperature=0.06,
            timeout=300,  # 5 minutes
        )
        self.chatLLM = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url_local,
            temperature=0.5,
            timeout=300,  # 5 minutes
            stop_sequences=["\n\n"],
        )
        self._initialized = True

    def get_fql_llm(self):
        return self.fqlLLM

    def get_chat_llm(self):
        return self.chatLLM
