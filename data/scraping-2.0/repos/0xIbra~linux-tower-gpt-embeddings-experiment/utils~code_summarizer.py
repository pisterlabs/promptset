import openai


class CodeSummarizer:

    @staticmethod
    def role_description():
        return """
        Summarize briefly without going too much into details the important part of given python code.
        """

    @staticmethod
    def summarize(code: str, model: str = 'gpt-3.5-turbo-0613'):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {'role': 'system', 'content': CodeSummarizer.role_description()},
                {'role': 'user', 'content': code}
            ],
            temperature=0.1
        )

        return response['choices'][0]['message']['content']
