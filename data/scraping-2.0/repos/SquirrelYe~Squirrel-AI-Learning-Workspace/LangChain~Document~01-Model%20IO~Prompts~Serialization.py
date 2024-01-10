from langchain.prompts import load_prompt

# ############### PromptTemplate ###############

def PromptTemplateDemo():
    basePromptTemplateDirectory = "./serialization-files/PromptTemplate/"

    prompt = load_prompt(basePromptTemplateDirectory + "simple_prompt.yaml")
    print(prompt.format(adjective="funny", content="chickens"))

    prompt = load_prompt(basePromptTemplateDirectory + "simple_prompt.json")
    print(prompt.format(adjective="funny", content="chickens"))

    prompt = load_prompt(basePromptTemplateDirectory + "simple_prompt_with_template_file.json")
    print(prompt.format(adjective="funny", content="chickens"))



# ############### FewShotPromptTemplate ###############

def FewShotPromptTemplateDemo():
    basePromptTemplateDirectory = "./serialization-files/FewShotPromptTemplate/"

    # Examples
    prompt = load_prompt(basePromptTemplateDirectory + "few_shot_prompt.yaml")
    # print(prompt.format(adjective="funny"))
    
    # Loading from YAML
    prompt = load_prompt(basePromptTemplateDirectory + "few_shot_prompt_yaml_examples.yaml")
    # print(prompt.format(adjective="funny"))
    
    # Loading from JSON
    prompt = load_prompt(basePromptTemplateDirectory + "few_shot_prompt.json")
    # print(prompt.format(adjective="funny"))
    
    # Examples in the Config File
    prompt = load_prompt(basePromptTemplateDirectory + "few_shot_prompt_examples_in.json")
    # print(prompt.format(adjective="funny"))
    
    # Example Prompt from a File
    prompt = load_prompt(basePromptTemplateDirectory + "few_shot_prompt_example_prompt.json")
    # print(prompt.format(adjective="funny"))


# ############### PromptTempalte with OutputParser ###############

def PromptTemplateWithOutputParserDemo():
    basePromptTemplateDirectory = "./serialization-files/PromptTempalte with OutputParser/"

    prompt = load_prompt(basePromptTemplateDirectory + "prompt_with_output_parser.json")
    temp = prompt.output_parser.parse("George Washington was born in 1732 and died in 1799.\nScore: 1/2")
    print(temp)
    # {'answer': 'George Washington was born in 1732 and died in 1799.', 'score': '1/2'}

if __name__ == "__main__":
    # PromptTemplateDemo()
    # FewShotPromptTemplateDemo()
    PromptTemplateWithOutputParserDemo()