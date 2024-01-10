import openai

openai.api_key = str(open("token.txt").read())

class StandardEnglish():
    def __init__(self, text_to_convert):
        self.text_to_convert = text_to_convert

    def convertStandardEnglish(self):

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Correct this to standard English:\n{self.text_to_convert}",
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        result = {
            'id': response.id,
            'created': response.created,
            'model': response.model,
            'completion_tokens': response.usage.completion_tokens,
            'prompt_tokens':response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
            'output': response.choices[0].text,
            'status': response.choices[0].finish_reason
        }

        return result["output"]