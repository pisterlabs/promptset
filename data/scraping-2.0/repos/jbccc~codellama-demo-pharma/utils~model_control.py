sys_test_prompt = (
    "You are a helpful asssistant working in Pharma and Clinical trials. "
    "You must write concise answer that takes no more than 200 tokens. "
)

def get_model_test_prompt_answer(prompt, llm_name, code_llm_name, temp, api_key, api_base):
    import openai as model
    # clears chat history
    model.ChatCompletion

    model.api_base = api_base  
    model.api_key = api_key

    return model.ChatCompletion.create(
        model=llm_name,
        messages=[{
                "role":"system", 
                "content": (sys_test_prompt)
            }, 
            {
                "role":"user", 
                "content": (prompt)
            }],
        temperature=temp,
    )

def get_model(api_conf):
    import openai as model
    model.api_base = api_conf["api_base"]
    model.api_key = api_conf["api_key"]
    return lambda system, user, llm, temp: model.ChatCompletion.create(
        model=llm,
        messages=[{
                "role":"system", 
                "content": (system)
            }, 
            {
                "role":"user", 
                "content": (user)
            }],
        temperature=temp,
    )