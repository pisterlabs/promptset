import openai
import vertexai
from vertexai.language_models import ChatModel

from api.cache.redis_helper import RedisHelper


class OpenAIChat:
    def __init__(self, AppSettings):
        self.model = AppSettings.openai_model
        self.max_token = AppSettings.openai_max_token
        self.temperature = AppSettings.openai_temperature
        self.redis_port = AppSettings.redis_port
        self.host = AppSettings.host
        openai.api_key = AppSettings.OPENAI_API_KEY

    def get_response_script(self, session_id: str, input_message: str) -> str:
        redis_helper = RedisHelper(host=self.host, port=self.redis_port)

        # Check if a response for this session_id already exists in Redis
        prev_response = redis_helper.get(session_id)

        messages = [
            {
                "role": "system",
                "content": """You are a customer service representative of Google Cloud Platform's Vertex AI. 
                            You only provide answers about Google Cloud Platform's Vertex AI and its products. 
                            You must provide very short answers. 
                            At the conclusion of your responses, always prompt the user to continue the conversation.""",
            },
            {"role": "user", "content": input_message},
        ]

        # Append the previous response to the messages if it exists
        if prev_response:
            messages.append({"role": "assistant", "content": prev_response})

        response = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=self.max_token,
            temperature=self.temperature,
            messages=messages,
        )
        response = response["choices"][0]["message"]["content"]

        # Store the latest response in Redis
        redis_helper.set(session_id, response)

        # If a previous response exists, print the latest response (optional, based on your requirements)
        if prev_response:
            redis_helper.print_latest_content(session_id)

        return response


class VertexAIChat:
    def __init__(self, AppSettings):
        vertexai.init(
            project=AppSettings.GCP_PROJECT_ID, location=AppSettings.VERTEXAI_LOCATION
        )
        self.chat_model = ChatModel.from_pretrained(AppSettings.pretrained_model_name)
        self.parameters = {
            "candidate_count": AppSettings.vertexai_candidate_count,
            "max_output_tokens": AppSettings.vertexai_max_token,
            "temperature": AppSettings.vertexai_temperature,
            "top_p": AppSettings.vertexai_top_p,
            "top_k": AppSettings.vertexai_top_k,
        }
        self.context = """You are a customer service representative of Google Cloud Platform's Vertex AI. 
        You only provide answers about Google Cloud Platform's Vertex AI and its products. 
        You must provide very short answers. 
        At the conclusion of your responses, always prompt the user to continue the conversation.
        """
        self.redis_port = AppSettings.redis_port
        self.host = AppSettings.host

    def get_response_script(self, session_id: str, input_message: str) -> str:
        redis_helper = RedisHelper(host=self.host, port=self.redis_port)

        # Check if a response for this session_id already exists in Redis
        prev_response = redis_helper.get(session_id)

        # Start the chat model session
        chat = self.chat_model.start_chat(context=self.context)

        # Get the response from the model
        response_obj = chat.send_message(input_message, **self.parameters)
        response = response_obj.text

        # Store the latest response in Redis
        redis_helper.set(session_id, response)

        # If a previous response exists, print the latest response (optional, based on your requirements)
        if prev_response:
            redis_helper.print_latest_content(session_id)

        return response
