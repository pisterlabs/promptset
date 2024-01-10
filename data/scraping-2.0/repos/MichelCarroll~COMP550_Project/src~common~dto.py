
from pydantic import BaseModel
from typing import Optional, Union
from openai.types.chat.completion_create_params import Function, FunctionCall

class TrainingExampleVariables(BaseModel):
    system_prompt: str 
    user_message: str 
    correct_response: str
    function_call: Optional[FunctionCall] = None 
    functions: list[Function] = []
    metadata: dict[str, str]
    
class TransformerFineTuningTrainingExample(BaseModel):
    serialized_completion: str
    correct_response: str

class OpenAIRoleMessage(BaseModel):
    role: str 
    content: str 

class OpenAIFunctionCall(BaseModel):
    name: str 
    arguments: str 

class OpenAIFunctionCallMessage(BaseModel):
    role: str  
    function_call: OpenAIFunctionCall 
    content: str = None

class OpenAIFineTuningTrainingExample(BaseModel):
    messages: list[Union[OpenAIRoleMessage, OpenAIFunctionCallMessage]]

class OpenAIFineTuningValidationExample(BaseModel):
    messages: list[Union[OpenAIRoleMessage, OpenAIFunctionCallMessage]]
    functions: Optional[list[dict]] = []
    function_call: Optional[str] = None

class FineTuningParameters(BaseModel):
    adapter_id_prefix: str
    base_model_name: str
    lora_rank: int = 64
    lora_alpha: int  = 16
    epochs: int = 1
    start_from_checkmarks: bool = False

    def checkmark_dir(self):
        return f"{self.adapter_id_prefix}-{self.lora_rank}-{self.lora_alpha}"

    def adapter_id(self):
        return f"{self.adapter_id_prefix}-r{self.lora_rank}-a{self.lora_alpha}-e{self.epochs}"