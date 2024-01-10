import os

class LLMAgent:

    def get_prompt(self, **kwargs):
        return ''

    def evaluation(self, **kwargs):
        inputs = kwargs.get('inputs')
        prompt = self.get_prompt(**inputs)
        import openai
        model_name =  kwargs.get('model_name')
        key = kwargs.get('key', '')
        temperature = kwargs.get('temperature', 0)
        if not os.getenv("OPENAI_API_KEY"):
            openai.api_key = key
        if openai.Model.retrieve(model_name):
            completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            response = completion.choices[0].message['content']
            return response

        else:
            raise ValueError(f'Do not have the access of {model_name}')
