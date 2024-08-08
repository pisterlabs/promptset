import hf

print("Loading LLM models...", flush=True)
llm = hf.LLM(["llama3-8b", "llama3-70b"])
print("LLM models loaded.", flush=True)

async def run_llm_coroutine(prompts, temperature=0.0, model="llama3-8b"):
    """
    Run the LLM model with the given prompts and temperature. 
    Input: List of prompts, temperature. Output: List of responses.
    """
    responses = []
    for p in prompts:
        responses.append(llm.generate_response(p, model=model, temperature=temperature))
    return responses