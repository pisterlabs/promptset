import torch
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any, Union

from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, GenerationConfig
from pydantic import Extra

class WizardLM(LLM, extra = Extra.allow):
    
    # HuggingFace models
    base_model: str = "TheBloke/wizardLM-7B-HF"
    #base_model: str = "ehartford/WizardLM-13B-Uncensored"
    
    # Model parameters
    temperature:float = 0
    num_beams:int = 6
    max_new_tokens:int = 512
    
    # Program parameters
    verbose:bool = False
    
    def __init__(self):
        super().__init__()
        
        # load the pipeline
        #tokenizer = LlamaTokenizer.from_pretrained(self.base_model, use_fast = True)
        #llm_model = LlamaForCausalLM.from_pretrained(self.base_model, torch_dtype = torch.float16, device_map = "auto")
        #llm_model.tie_weights()
        
        self._pipe = pipeline(task = "text-generation", model = self.base_model, tokenizer = self.base_model,
                              device_map = "auto", torch_dtype = torch.float16)
        
        # Generation configuration
        self._pipe.tokenizer.truncation_side = 'left'
        self.generation_configs = GenerationConfig.from_pretrained(
            pretrained_model_name = self.base_model, temperature = self.temperature, num_beams = self.num_beams, max_new_tokens = self.max_new_tokens, top_p = .85)
        
        #       num_beam_groups = 2

        self._device = self._pipe.device
        
        print(f"\n\n[INFO] {self.base_model} ({self._pipe.torch_dtype}) has been loaded! "\
            f"({self._device}, {torch.cuda.get_device_name(self._device)})\n")
    
    def __evaluate(self, prompts: Union[List, str]):

        # Generate the output
        outputs = self._pipe(prompts, return_full_text = False, clean_up_tokenization_spaces = True, batch_size = 100, 
                             generation_config = self.generation_configs)

        # Extract the generated text
        outputs = [output['generated_text'].strip() for output in outputs] 
        
        # Extract the textual data
        if len(outputs) == 1:
            return outputs[0]
        
        return outputs
    
    def _call(self, prompts: Union[List, str], stop: Optional[List[str]] = None) -> str: 
        return self.__evaluate(prompts)  # type: ignore
    
    @property
    def _llm_type(self) -> str:
        return self.base_model
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        
        model_params = {"base_model": self.base_model}
        model_params.update(self.generation_configs.to_dict())
        return model_params
