import openai

class OpenaiClient:
    def run_prompt(prompt_):
        model_engine = "text-davinci-003"
        openai.api_key = 'XXX--REDACTED--XXX'
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt_,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0,
        )
        return completion.choices[0].text