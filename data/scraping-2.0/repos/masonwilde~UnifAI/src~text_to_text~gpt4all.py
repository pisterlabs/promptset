from langchain.llms import GPT4All


def get_llm(config):
    llm = GPT4All(
        model=config.llm_model,
    )
    return llm


class TTT:
    def __init__(self, config):
        self.llm = get_llm(config)
        self.config = config

    def prompt(self, prompt):
        return self.llm(prompt)
