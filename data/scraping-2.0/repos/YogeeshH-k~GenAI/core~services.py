import openai
from django.conf import settings

OPEN_AI_API_KEY = settings.OPEN_AI_API_KEY
OPEN_AI_ORGANIZATION_ID = settings.OPEN_AI_ORGANIZATION_ID


class OpenAI:
    def __init__(self):
        openai.api_key = OPEN_AI_API_KEY
        openai.organization = OPEN_AI_ORGANIZATION_ID

    def get_completion(self, prompt, engine="davinci", max_tokens=150, temperature=0.9, top_p=1, frequency_penalty=0.0,
                       presence_penalty=0.6, stop=["\n"]):
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
        return response

    def fine_tune(self, file, model="davinci", n_epochs=3, batch_size=64, learning_rate=1e-5, validation_split=0.1,
                  save_every=200, save_path=""):
        response = openai.FineTune.create(
            file=file,
            model=model,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
            save_every=save_every,
            save_path=save_path
        )
        return response

    def embeddings(self, documents, model="ada"):
        response = openai.Embedding.create(
            documents=documents,
            model=model
        )
        return response
