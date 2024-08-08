from hf import LLM

if __name__ == "__main__":
    # llm = LLM(["gemma-2b"])
    llm = LLM(["gemma-2b", "llama3-8b", "llama3-70b"])
    prompt = "What is the meaning of life?"
    print(f"Q. Gemma-2b; {prompt}; A. {llm.generate_response(prompt, model='gemma-2b')}")
    print(f"Q. Llama3-8b; {prompt}; A. {llm.generate_response(prompt, model='llama3-8b')}")
    print(f"Q. Llama3-70b; {prompt}; A. {llm.generate_response(prompt, model='llama3-70b')}")