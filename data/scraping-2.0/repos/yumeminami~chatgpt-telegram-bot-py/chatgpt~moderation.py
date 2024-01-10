import openai
import os


def moeradtions(input: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.api_key ="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"
    try:
        response = openai.Moderation.create(
            input=input,
        )
        print(response)
        return response.results[0]["flagged"]
    except Exception as e:
        print(e)
        return False
