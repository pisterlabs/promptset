import config
import openai

class StaticOpenAIModel:

    @staticmethod
    def generate_response(messages):
        #config.logger.debug(f"Entering generate_response with messages: {messages}")
        response_obj = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
        )

        return response_obj

    @staticmethod
    def generate_embedding(text):
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        embedding = response.data[0].embedding
        return embedding
