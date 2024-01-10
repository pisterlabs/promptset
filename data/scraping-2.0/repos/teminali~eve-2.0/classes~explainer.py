import json
import openai


class Explainer:
    def __init__(self):
        # open config.json file and load api key
        with open("services/config.json") as f:
            config = json.load(f)
            key = "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"

        openai.api_key = key
        self.start_sequence = "\nAI:"
        self.restart_sequence = "\nHuman: "

    @staticmethod
    def generate_response(prompt=None, question=None, temperature=0.9, max_tokens=150, top_p=1, frequency_penalty=0,
                          presence_penalty=0.6, ):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=[" Human:", " AI:"]
        )
        return response.get('choices')[0].get('text').strip()

# example on using the class
# if __name__ == "__main__":
#     explainer = GetMessage()
#     print(explainer.generate_response(prompt="", question=""))
