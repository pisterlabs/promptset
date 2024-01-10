from langchain.schema import HumanMessage
# ----------------------------------------------------------------------------------------------------------------------
import tools_LLM_Azure
import tools_LLM_OPENAI
# ----------------------------------------------------------------------------------------------------------------------
def ex1(prompt):
    tools_LLM_OPENAI.LLM('./secrets/openaiapikey_private_D.txt')
    response = tools_LLM_OPENAI.gpt3_completion(prompt)
    print(response)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex2(prompt):
    chatmode=True

    LLM = tools_LLM_Azure.LLM('./secrets/private_config_azure.yaml',chatmode=True)
    if chatmode:
        prompt = [HumanMessage(content=prompt)]

    response = LLM(prompt)
    print(response.content)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    prompt = 'Who framed Roger rabit?'
    ex2(prompt)

