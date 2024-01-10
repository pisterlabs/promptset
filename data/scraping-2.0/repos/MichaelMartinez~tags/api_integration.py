# api_integration.py
from gpt4all import GPT4All
import openai

class APIIntegration:
    def send_file(self, file):
        # Code to send the file to the API for processing and receive the response
        # Replace this with the actual API integration code
        response = f"Processed text for {file}"
        return response
    
    def test_api(self, string):
        model = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")
        output = model.generate(string, max_tokens=20)
        print(output)
        return output
    
    def test_gpt_openai(self, string):
        openai.api_base = "http://localhost:4891/v1"
        #openai.api_base = "https://api.openai.com/v1"

        openai.api_key = "not needed for a local LLM"

        # Set up the prompt and other parameters for the API request
        prompt = string

        # model = "gpt-3.5-turbo"
        #model = "mpt-7b-chat"
        model = "ggml-model-gpt4all-falcon-q4_0.bin"

        # Make the API request
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=50,
            temperature=0.28,
            top_p=0.95,
            n=1,
            echo=True,
            stream=False
        )

        # Print the generated completion
        print(response)