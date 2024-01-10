from langchain.prompts import (
    PromptTemplate,
    load_prompt
)


class PromptFactory():
    PROMPTDIR: str = "./templates/prompts"

    prompts: dict = {}

    def __init__(self):
        self.prompts["base_prompt"] = load_prompt(f"{self.PROMPTDIR}/base.json")
        pass

    def append_prompt(self, prompt: PromptTemplate) -> None:
        pass

    def remove_prompt(self, prompt_key: str) -> None:
        pass

    def get_prompt_keys(self) -> [str]:
        return list(self.prompts.keys())

    def get_prompt_dict(self) -> dict:
        return self.prompts
    
    def get_prompts(self) -> [PromptTemplate]:
        return list(self.prompts.values())
    