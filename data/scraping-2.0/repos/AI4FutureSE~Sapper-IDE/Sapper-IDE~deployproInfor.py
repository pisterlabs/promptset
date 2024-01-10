import openai

# openai.api_key = "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"

def program_Generate(prompt, num_candidates=1, max_tokens=256, stop=None, temperature=0):
    results = []
    try:
        response = openai.Completion.create(
            prompt=prompt,
            model="text-davinci-003",
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            n=num_candidates
        )
        return response["choices"][0]["text"]
    except Exception as e:
        print(type(e), e)
        if str(type(e)) == "<class 'openai.error.InvalidRequestError'>":
            response = "null"
    return results


question_prompt = """A developer is crafting prompts and giving them to the LLM in order to get the model to complete a task or generate output as an AI service.
Here's a Prompt about the small AI service. 
Please understand the task completed by Prompt, and then write a pre-message to the task to remind the user to use the service. 
This pre-information should include a description of the AI service and what the user needs to input for the first time, and written in the first person
Prompts: {{Prompts}}
pre-information:
"""
def generate_deploypreInfor(query, OpenAIKey):
    openai.api_key = OpenAIKey
    # TODO figure out alternative stopping criterion for generating initial characters?
    question_prompt1 = question_prompt.replace("{{Prompts}}", query)
    if len(question_prompt1) >=2000 :
        question_prompt1 = question_prompt1[0:2000]
    expansion = program_Generate(prompt=question_prompt1, temperature=0.7, max_tokens=512, num_candidates=1)
    return expansion



