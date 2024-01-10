from dataclasses import dataclass
from datetime import datetime
import traceback
from typing import List
import openai
import torch
from src.constants import OpenAIRoles
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM
from src.constants import openai_system_message
from yaspin import yaspin

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
device_map = {
    "": "cpu",
}

# print("Loading tokenizer for chat bot...")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

# would "decapoda-research/llama-30b-hf" work?
print("Loading tokenizer for chat bot...")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="EleutherAI/gpt-neox-20b")
print("Loading chat bot...")
chatbot = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neox-20b",
    load_in_8bit=True,
    device_map=device_map,
    quantization_config=quantization_config,
)

pipe = pipeline(
    "text-generation",
    model=chatbot, 
    tokenizer=tokenizer, 
    max_length=2048,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

# local_llm = HuggingFacePipeline(pipeline=pipe) # lang chain

system_message = {
    "role": "system",
    "content": openai_system_message,
}

TIMEOUT_SECS = 60
MESSAGES_COUNT_LIMIT = 4000

@dataclass
class MessageRepresentation:
    content: str
    role: OpenAIRoles
    tokens_count: int

    def __init__(self, content: str, role: OpenAIRoles, tokens_count: int = 0):
        self.content = content
        self.role = role
        if tokens_count == 0:
            self.tokens_count = count_message_tokens(content)
        self.tokens_count = tokens_count
    
    def to_msg(self):
        return {
            "role": self.role.name,
            "content": self.content
        }
    
    def to_text_completion_msg(self):
        return self.role.name + ": " + self.content

def count_message_tokens(message: str) -> int:
    # input_ids = torch.tensor(tokenizer.encode(message)).unsqueeze(0)
    # num_tokens = input_ids.shape[1]
    # return num_tokens
    return 0

total_history_tokens_count = count_message_tokens(system_message["content"])

def add_message_to_history(message: str, role: OpenAIRoles, messages_history: List[MessageRepresentation]):
    global total_history_tokens_count

    message_tokens_count = count_message_tokens(message)

    messages_history.append(MessageRepresentation(
        content = message,
        role = role,
        tokens_count = message_tokens_count
    ))
    total_history_tokens_count += message_tokens_count

def format_messages_into_text_completion_request(messages: List[MessageRepresentation]) -> str:
    text = '''You are AI assistant that is using
[retrieve], [store] and [delete] plugins to make better conversations with user and manage AI assistants own memory,
[google] plaugin to search internet,
[open] plaugin to read content summary from urls,
[time] plugin to read current user time.
If you dont not know the answer to a question, truthfully say you do not know.
Below is the record of our conversation:

{history}
assistant:'''
    history = "\n".join([message.to_text_completion_msg() for message in messages[-10:]])

    return text.format(history=history)

@yaspin(text="Processing...", color="white", spinner="dots", attrs={"bold": True})
def send_messages_history_to_open_ai(messages_history: List[MessageRepresentation], model) -> str:
    global total_history_tokens_count

    while True:
        if total_history_tokens_count > MESSAGES_COUNT_LIMIT:
            # pop both chatbot and user messages
            total_history_tokens_count -= messages_history[0].tokens_count
            messages_history.pop(0)
            if len(messages_history) > 0:
                total_history_tokens_count -= messages_history[0].tokens_count
                messages_history.pop(0)
        else:
            break

        if len(messages_history) == 0:
            raise Exception("Error: Could not create chat completion. Messages history is empty.")
    
    messages = [system_message]
    for message_from_history in messages_history:
        messages.append({
            "role": message_from_history.role.name,
            "content": message_from_history.content
        })
    
    try:
        # completion = openai.ChatCompletion.create(
        #     model=model,
        #     max_tokens=500,
        #     temperature=0.7,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0.6,
        #     timeout=TIMEOUT_SECS,
        #     messages=messages
        # )

        # return completion.choices[0].message.content # type: ignore
        generation = pipe(format_messages_into_text_completion_request(messages_history))
        print("--- DEBUG:")
        print(generation)
        return "".join(str(filter(lambda x: len(x) > 1, generation[0]["generated_text"].split("\n"))[-1]).split("assistant:", maxsplit=2)[1]) # type: ignore

    except Exception as e:
        global retries_count
        print(e)
        print("Error: Could not create chat completion.")
        traceback.print_exc()
        return ""
