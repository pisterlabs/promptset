from typing import List, Dict, Optional
from functions import Function
from .index import FunctionIndex
from utils import Logger
import openai

class IntentIndex(FunctionIndex):

    functions: Dict[str, Function]
    _logger: Optional[Logger]
    
    def __init__(self, logger: Optional[Logger] = None):
        self.functions = {}
        self._logger = logger

    def put(self, function: Function):
        if not isinstance(function, Function):
            raise TypeError('CREATE ERROR: function must be of type Function')
        
        if function.name not in self.functions:
            self.functions[function.name] = function

    def get(self, name: str) -> Function:
        return self.functions.get(name)

    def retrieve(self, thought: str, k: int = 10) -> List[Function]:
        
        _PROMPT = '''Given the intent of a system, rate the relevancy of a function in helping solve this task on a scale of 0 to 5.\n\nINTENT: I need to raise 55 to the power of 0.12\nFUNCTION: order_coffee\nDESCRIPTION: order coffee through Rose's coffee shop. Send post request to http://3.133.95.18/order with form data input:<coffee_type> for <your name>\nRELEVANCE: 0\n\nINTENT: I need to find out joe rogan's current age\nFUNCTION: google_search\nDESCRIPTION: Enter a question you would like to know. Returns text answers from the internet.\nRELEVANCE: 4\n\nINTENT: I need to find out joe rogan's birthday\nFUNCTION: google_search\nDESCRIPTION: Enter a question you would like to know. Returns text answers from the internet.\nRELEVANCE: 5\n\nINTENT: I need to find out the author of "the art of war"\nFUNCTION: google_search\nDESCRIPTION: Enter a question you would like to know. Returns text answers from the internet.\nRELEVANCE: 5\n\nINTENT: {thought}\nFUNCTION: {function_name}\nDESCRIPTION: {function_description}\nRELEVANCE:'''

        function_relevance = {}

        for function_name, function in self.functions.items():
            prompt = _PROMPT.format(
                thought=thought, 
                function_name=function_name, 
                function_description=function.description
            )
            
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=1,
            )

            relevancy = int(response['choices'][0]['text'].strip())
            function_relevance[function_name] = relevancy

        # Sort functions by relevance and get top k
        sorted_functions = sorted(function_relevance.items(), key=lambda x: x[1], reverse=True)
        top_k_functions = [self.get(fn_name) for fn_name, _ in sorted_functions[:k]]

        # If logger is provided, log the relevancy scores
        if self._logger is not None:
            function_str = ''
            for fn_name, relevancy in sorted_functions[:k]:
                function_str += f'\n{fn_name}: {relevancy}'
            self._logger.log(function_str, category='RELEVANT')

        return top_k_functions
