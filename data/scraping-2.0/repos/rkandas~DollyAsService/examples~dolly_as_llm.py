# Prerequisites: pip install transformers,langchain, torch
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DollyLLM(LLM):
    
    temperature: float = 0.95
    max_tokens: int = 64
    do_sampling: bool = False
    top_k: int = 50
    model_path:str = "databricks/dolly-v2-2-8b"
    END_KEY = "### End"
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer

    def sanitize_and_tokenize(self, text: str, max_tokens: int = 512):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        return tokens
    
    # Create the model instance and load the weights
    def __init__(self, temperature: float = 0.95, max_tokens: int = 64, do_sampling: bool = False, top_k: int = 50):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.do_sampling = do_sampling
        self.top_k = top_k
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, pad_token_id=self.tokenizer.eos_token_id, torch_dtype=torch.float16)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        
    @property
    def _llm_type(self) -> str:
        return "Dolly_v2_3B"
    
    # This method will be called by Langchain to generate a response
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            input_text = prompt #f"### Instruction:\n{prompt.prompt}\n\n### Response:\n"
            tokens = self.sanitize_and_tokenize(input_text)
            tokens.to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**tokens,  max_length=self.max_tokens, do_sample=self.do_sample,temperature=self.temperature,top_k=self.top_k, pad_token_id=self.tokenizer.eos_token_id, torch_dtype=torch.float16)
            response = self.tokenizer.decode(output[0], skip_special_tokens=False)
            response = response.split("### Response:\n")[-1]
            
            stop_index = -1
            if stop:
                for s in stop:
                    if s in response:
                        stop_index = response.find(s)
                        break

            if stop_index == -1:
                return {"response": response}
            else:
                return {"response": response[:stop_index]}
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}")

    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "do_sampling": self.do_sampling, "top_k": self.top_km, "model_path": self.model_path}