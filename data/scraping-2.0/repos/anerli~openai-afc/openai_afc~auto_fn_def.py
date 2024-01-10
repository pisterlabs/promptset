from openai_afc import AutoFnParam
from typing import Callable, List, Any

class AutoFnDefinition:
    def __init__(self, fn: Callable, name: str = None, description: str = None, params: List[AutoFnParam] = None, output_transform : Callable[[Any], str]=str):
        '''
        fn: Function to be auto-called.
        name: Name of function to pass to GPT. Defaults to fn.__name__
        description: Description of function to pass to GPT
        params: List of param objects
        output_transform: How to transform results from functions before given back to GPT. Must be a function that returns a string.
        '''
        self.fn = fn
        if name == None:
            self.name = fn.__name__
        self.description = description
        if params is None:
            self.params = []
        else:
            self.params = params
        self.output_transform = output_transform
    
    def get_metadata(self) -> dict:
        '''
        Get function metadata which can be passed to OpenAI API
        '''
        required = [p.name for p in self.params if p.required]
        properties = {p.name: p.schema for p in self.params}
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required
            }
        }