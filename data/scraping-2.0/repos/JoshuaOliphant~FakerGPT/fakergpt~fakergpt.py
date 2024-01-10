import openai
import os


class GPTFaker:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.llm_name = "gpt-3.5-turbo"

    def call_gpt(self, prompt):

        # Make the API call
        response = openai.ChatCompletion.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": f"{prompt}"}],
            temperature=0.8
        )

        # Extract the generated text from the API response
        response_text = response.choices[0].message.content

        return response_text


if __name__ == "__main__":
    faker = GPTFaker()
    print(faker.call_gpt("Generate a fake name"))
