"""
TextualEntailmentPrompter.py
"""
import openai

class TextualEntailmentPrompter:
    def __init__(self, openai_api_key: str, engine: str = "text-davinci-002"):
        self.openai_api_key = openai_api_key
        self.engine = engine


    def predict(self, premise: str, hypothesis: str):
        prompt = f"""
        Does the premise entail, contradict, or have no relationship to the hypothesis? Label the sentence pair as contradiction, entailment, or neutral.
        ###
        premise: {premise}
        hypothesis: {hypothesis}
        ###
        Label:
        """

        openai.api_key = self.openai_api_key
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response.choices[0].text.strip().lower()