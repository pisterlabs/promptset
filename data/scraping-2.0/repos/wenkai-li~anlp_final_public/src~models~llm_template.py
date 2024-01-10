from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# given the system, user and ai prompt, return the generate prompt for llama2
def generate_llama_prompt(system_prompt, user_prompts: list, ai_prompts: list):
    # check the size of user-ai prompt pair
    assert(len(user_prompts)==len(ai_prompts)+1)
    whole_prompt = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n".format(system_prompt)
    if len(ai_prompts)==0:
        whole_prompt += user_prompts[0]
        whole_prompt += " [/INST]"
    else:
        for user_prompt, ai_prompt in zip(user_prompts[:-1], ai_prompts):
            whole_prompt += user_prompt
            whole_prompt += " [/INST]"
            whole_prompt += ai_prompt
            whole_prompt += "</s><s> [INST] \n"
        whole_prompt += user_prompts[-1]
        whole_prompt += " [/INST]"
    return whole_prompt

def generate_gpt_prompt(system_prompt, user_prompts: list, ai_prompts: list):
    assert(len(user_prompts)==len(ai_prompts)+1)
    whole_prompt = [SystemMessagePromptTemplate.from_template(system_prompt)]
    if len(ai_prompts)==0:
        whole_prompt.append(HumanMessagePromptTemplate.from_template(user_prompts[0]))
    else:
        for user_prompt, ai_prompt in zip(user_prompts[:-1], ai_prompts):
            whole_prompt.append(HumanMessagePromptTemplate.from_template(user_prompt))
            whole_prompt.append(AIMessagePromptTemplate.from_template(ai_prompt))
        whole_prompt.append(HumanMessagePromptTemplate.from_template(user_prompts[-1]))
    return whole_prompt