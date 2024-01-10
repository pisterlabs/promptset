from typing import Dict
from langchain.llms.base import LLM
from transformers import T5ForConditionalGeneration


class CustomHFModel(LLM):
    tokenizer: object
    model: object
    generate_kwargs: Dict
    
    @property
    def _llm_type(self) -> str:
        return "custom"
        
    def _call(self, prompt, stop=None):
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
            
        inputs = self.tokenizer(prompt, padding=False, return_tensors="pt").to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        generated_sequence = self.model.generate(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 pad_token_id=self.tokenizer.eos_token_id,
                                                 **self.generate_kwargs)
        
        text = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
        # T5 models return just the generated text (HF calls it text2text-generation)
        # while CausalLM like GPT return the prompt and the generated text (HF calls it text-generation)
        # so we need to remove the prompt from the generated text
        if not isinstance(self.model, T5ForConditionalGeneration):
            text = text[len(prompt):]
        
        return text
