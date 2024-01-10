import tiktoken
from pydantic import BaseModel
from flamethrower.models.models import OPENAI_GPT_4_TURBO

class TokenCounter(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    pricing: dict = {
        'gpt-4-1106-preview': {
            'input': {
                'cost': 0.01,
                'unit': 'tokens',
                'per': 1000
            },
            'output': {
                'cost': 0.03,
                'unit': 'tokens',
                'per': 1000
            }
        }
    }

    def add_input_tokens(self, tokens: int) -> None:
        self.total_input_tokens += tokens
    
    def add_output_tokens(self, tokens: int) -> None:
        self.total_output_tokens += tokens
    
    def add_streaming_input_tokens(self, complete_input_text: str) -> None:
        # TODO: no hardcoded models
        enc = tiktoken.encoding_for_model(OPENAI_GPT_4_TURBO)
        num_input_tokens = len(enc.encode(complete_input_text))

        self.total_input_tokens += num_input_tokens
    
    def add_streaming_output_tokens(self, complete_output_text: str) -> None:
        # TODO: no hardcoded models
        enc = tiktoken.encoding_for_model(OPENAI_GPT_4_TURBO)
        num_output_tokens = len(enc.encode(complete_output_text))

        self.total_output_tokens += num_output_tokens
        
    def return_cost_analysis(self) -> str:
        input_cost = (
            self.total_input_tokens 
            * self.pricing['gpt-4-1106-preview']['input']['cost']
            / self.pricing['gpt-4-1106-preview']['input']['per']
        )
        output_cost = (
            self.total_output_tokens
            * self.pricing['gpt-4-1106-preview']['output']['cost']
            / self.pricing['gpt-4-1106-preview']['output']['per']
        )
        total_cost = input_cost + output_cost

        return (
            'Total tokens used:\n'
            f'  Input tokens: {self.total_input_tokens} => ${input_cost:.2f}\n'
            f'  Output tokens: {self.total_output_tokens} => ${output_cost:.2f}\n'
            f'  💸 Total cost: ${total_cost:.2f}'
        )
