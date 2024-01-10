import openai

class BaseExtractor():
    def __init__(
        self,
        template,
        engine: str = 'gpt-3.5-turbo',
    ):
        self.model = None
        self.engine = engine
        self.template = template

    def gpt_call(self, phrase):
        print(self.template + phrase)
        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=self.message_wrapper(self.template + phrase),
            max_tokens=100,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response
    
    def message_wrapper(self, prompt):
        return [{
            "role": "user",
            "content": prompt
        }]
    
    def parse_response(self, response):
        entities = response.choices[0].message.content
        return [
            self.strip(entity)
            for entity in entities.split('\n')
            if entity != ''
        ]
    
    def strip(self, entity):
        return entity[3:].split('(')[0]
    
    def extract(self, phrase):
        response = self.gpt_call(phrase)
        parsed = self.parse_response(response)
        # print(response)
        return parsed

    def set_gpt_call(self, gpt_call):
        self.gpt_call = gpt_call

class ColumnExtractor(BaseExtractor):
    def __init__(
        self,
        stat_type: str,
        engine: str = 'gpt-3.5-turbo',
    ):
        super().__init__(
            template=f'Extract all of the distinct {stat_type} from the following expression: ',
            engine=engine,
        )

class FuncExtractor(BaseExtractor):
    def __init__(
        self,
        engine: str = 'gpt-3.5-turbo',
    ):
        super().__init__(
            template='Extract all of the distinct functions from the following expression: ',
            engine=engine,
        )
