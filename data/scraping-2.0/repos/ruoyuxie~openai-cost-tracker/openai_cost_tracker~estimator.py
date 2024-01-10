import functools
from lorem_text import lorem
from .utils import num_tokens_from_messages
import openai

class CostEstimator:

    # Source: https://openai.com/pricing
    # Prices in $ per 1000 tokens
    # Last updated: 2023-12-6

    MODEL_INFO = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo-1106": {"input": 0.0015, "output": 0.002},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
    }

    total_cost = 0.0  # class variable to persist total_cost

    def __init__(self) -> None:
        self.default_model = "gpt-3.5-turbo-1106"

    @classmethod
    def reset(cls) -> None:
        cls.total_cost = 0.0

    def get_total_cost(self) -> float:
        return "{:.4f}".format(CostEstimator.total_cost).rstrip('0').rstrip('.')

    def __call__(self, function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            simulation = kwargs.get('simulation', True)
            model = kwargs.get('model', self.default_model)
            simulated_max_tokens = kwargs.get('simulated_max_tokens', kwargs['max_tokens'])
            simulated_max_tokens = int(simulated_max_tokens/2) # TODO: create a better token number simulation
            print_cost = kwargs.get('print_cost', kwargs['print_cost'])

            messages = kwargs.get("messages")
            input_tokens = num_tokens_from_messages(messages, model=model)

            if simulation:
                simulation_output_message_content = lorem.words(simulated_max_tokens) if simulated_max_tokens else "Default response"
                simulation_output_messages = {
                    'choices': [
                        {
                            'message': {
                                'content': simulation_output_message_content
                            }
                        }
                    ]
                }
                #output_tokens = num_tokens_from_messages([{"role": "assistant", "content": simulation_output_message_content}], model=model)

                # Assume that the number of simulated output tokens are the same as the max tokens 
                output_tokens = simulated_max_tokens 
                total_tokens = input_tokens + output_tokens
            else:
                response = function(*args, **kwargs)
                total_tokens = response.usage.total_tokens
                output_tokens = total_tokens - input_tokens

            input_cost = input_tokens * self.MODEL_INFO[model]['input'] / 1000
            output_cost = output_tokens * self.MODEL_INFO[model]['output'] / 1000
            cost = input_cost + output_cost
            CostEstimator.total_cost += cost  # update class variable

            if print_cost:
                print(f"\033[92mInput tokens: {input_tokens} | Output tokens: {output_tokens} | Cost: ${cost:.4f} | Total: ${CostEstimator.total_cost:.4f}\033[0m")

            return simulation_output_messages if simulation else response
        return wrapper
    
@CostEstimator()
def query_openai(model, messages, **kwargs):
    estimator_args = ['simulation', 'simulated_max_tokens', 'print_cost']

    for arg in estimator_args:
        if arg in kwargs:
            del kwargs[arg]

    return openai.ChatCompletion.create(
        model = model,
        messages = messages,
        **kwargs) 