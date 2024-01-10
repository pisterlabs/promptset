from openai import OpenAI, BadRequestError
from sklearn.metrics.pairwise import cosine_similarity
from functools import cache


class GPT:
    def __init__(
        self,
        api_key,
        multimodal_model="gpt-4-vision-preview",
        image_model="dall-e-2",
        prompt_model="gpt-4-1106-preview",
    ) -> None:
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.multimodal_model = multimodal_model
        self.image_model = image_model
        self.prompt_model = prompt_model

    @cache
    def get_embeddings(self, input: str, model: str = "text-embedding-ada-002"):
        embeddings = self.client.embeddings.create(input=input, model=model)
        return embeddings.data[0].embedding

    @cache
    def generate_image(self, prompt, size="1024x1024", quality="standard", n=1):
        for _ in range(0, 3):
            try:
                return self.client.images.generate(
                    model=self.image_model, prompt=prompt, size=size, quality=quality, n=n
                )
            except BadRequestError as error:
                print("Request failed. Retry. Error: ", error)
        raise Exception("Failed to generate image!")

    def get_image_url(self, image):
        return str(image.data[0].url)

    def detect_differences(
        self, initial_prompt, url, system_message, max_tokens=150, temperature=0.2
    ):
        generation = self.client.chat.completions.create(
            model=self.multimodal_model,
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": initial_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                            },
                        },
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            # top_p=0.3
        )
        return generation.choices[0].message.content

    def generate_prompt_from_differences(
        self, previous_messages, temperature, system_message
    ):
        messages = [{"role": "system", "content": system_message}]
        messages += previous_messages
        generation = self.client.chat.completions.create(
            model=self.prompt_model, messages=messages, temperature=temperature
        )
        return generation.choices[0].message.content
