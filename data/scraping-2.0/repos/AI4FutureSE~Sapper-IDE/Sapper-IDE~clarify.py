import openai

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
        for choice in response.choices:
            text = choice.text.strip()
            print(text)
            results.append(text.split(". You")[0])
    except Exception as e:
        print(type(e), e)
        if str(type(e)) == "<class 'openai.error.InvalidRequestError'>":
            response = "null"
    return results


question_prompt = """
Context: A user is interacting with a large language model.  They are crafting prompts and giving them to the LLM in order to get the model to complete a task or generate output.
Instruction: In order for a large language model to better complete tasks or generate outputs, need to ask a question about the task and let users reply.
The questions asked can be combined with the previously obtained Key Requirements, User Preference, and Implementing Consideration to make users more clear about their needs.
{{User_Behaviour}}
The questions asked need to lead users to clarification of requirements and conform to strategies for interacting with LLM.  

Functional requirement: I want to develop a service that automatically draws according to the weather.
Requirement guidance: You need to consider what goes into the design. For example, which colors to use for painting, canvas size, canvas type, etc.
Answer: Draw 500x500 pixel RGB color pictures.
Requirement guidance: You need to specify some conditions. For example, paint automatically only on rainy or sunny days.
Answer: Abstract pictures when the weather is rainy and nature landscapes when the weather is sunny.
Please give requirement guidance for the following functional requirement based on the above form.
"""

question_prompt2 = """
Context: A user is interacting with a large language model. They are crafting prompts and giving them to the LLM in order to get the model to complete a task or generate output.
You are a product manager AI tasked with focusing on users' requirements and understanding them through deep communication.
You need to provide a question about the users' requirements and let users reply.
The question raised can be referred to the user's task notes：
{{User_Behaviour}}
The question should be well in line with user requirements and usage scenarios.

Functional requirement: I want to develop a service that automatically draws according to the weather.
Requirement guidance: You need to consider what goes into the design. For example, which colors to use for painting, canvas size, canvas type, etc.
Answer: Draw 500x500 pixel RGB color pictures.
Requirement guidance: You need to specify some conditions. For example, paint automatically only on rainy or sunny days.
Answer: Abstract pictures when the weather is rainy and nature landscapes when the weather is sunny.
Please give requirement guidance for the following functional requirement based on the above form.
"""
def generate_query_expansion(Behaviour, query, OpenAIKey):
    openai.api_key = OpenAIKey
    # TODO figure out alternative stopping criterion for generating initial characters?
    question_prompt1 = question_prompt + query + 'Requirement guidance:'
    question_prompt1 = question_prompt1.replace("{{User_Behaviour}}", Behaviour)
    expansion = program_Generate(prompt=question_prompt1, temperature=0.3, max_tokens=48, num_candidates=1, stop='\n')[0]
    if "Requirement guidance:" in query:
        query = query + "\nWrite the Functional requirement in detail according to the Requirement guidance and Answer above\nFunctional Requirement:"
        expansion1 = program_Generate(prompt=query, num_candidates=1, max_tokens=256, temperature=0.3)[0]
    else:
        expansion1 = query.replace("Functional requirement:", "")

    return expansion, expansion1



