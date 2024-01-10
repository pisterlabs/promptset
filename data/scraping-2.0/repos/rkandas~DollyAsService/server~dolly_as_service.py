import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain import PromptTemplate
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

END_KEY = "### End"
model_path = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

template = "### Instruction:\n{instruction}\n\n### Response:\n"
prompt_template = PromptTemplate(template=template, input_variables=["instruction"])
class Prompt(BaseModel):
    prompt: str

def sanitize_and_tokenize(text: str, max_tokens: int = 512):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    return tokens

async def generate(prompt:str, temperature: float = 0.95, max_tokens: int = 128, top_k: int = 500, end_key: str = END_KEY, do_sample: bool = True):
    try:
        input_text = prompt
        tokens = sanitize_and_tokenize(input_text)
        tokens.to(model.device)
        with torch.no_grad():
            output = model.generate(**tokens,  max_length=max_tokens, do_sample=do_sample,temperature=temperature,top_k=top_k)
        response = tokenizer.decode(output[0], skip_special_tokens=False)
        response = response.split("### Response:\n")[-1]

        stop_index = response.find(end_key)
        if stop_index == -1:
            return {"response": response}
        else:
            return {"response": response[:stop_index]}
    except Exception as e:
        error_message = f"Error: {str(e)}\n"
        stack_trace = traceback.format_exc()
        print( error_message + stack_trace )

@app.post("/api/instruct_generate")
async def instruct_generate_text(prompt: Prompt, temperature: float = 0.95, max_tokens: int = 128, top_k: int = 500, end_key: str = END_KEY, do_sample: bool = True):
        input_text = prompt_template.format(instruction=prompt.prompt)
        generated = await generate(input_text, temperature, max_tokens, top_k, end_key, do_sample)
        return generated
    
@app.post("/api/freeform_generate")
async def freeform_generate_text(prompt: Prompt, temperature: float = 0.95, max_tokens: int = 128, top_k: int = 500, end_key: str = END_KEY, do_sample: bool = True):
        input_text = prompt.prompt
        generated = await generate(input_text, temperature, max_tokens, top_k, end_key, do_sample)
        return generated

@app.on_event("shutdown")
def shutdown_event():
    # Unload the model and perform any required cleanup
    global model
    del model
    torch.cuda.empty_cache()
    print("Model unloaded and resources cleaned up")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
