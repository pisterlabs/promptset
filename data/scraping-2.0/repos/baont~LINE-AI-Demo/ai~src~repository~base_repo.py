from langchain.prompts import PromptTemplate


class BaseRepo:
    @staticmethod
    def load_messages(config: dict, key: str):
        """Load the template in the config into chat messages
        The Human message is templated as {input}

        Example:

        ```yaml
        ExamplePrompt1:
            prompt: >
                This is a template
        ```
        """
        c = config[key]
        prompt = c.get("prompt")

        if isinstance(prompt, str):
            prompt_str = prompt + "\n[INST] {input} [/INST]\n"
            return PromptTemplate.from_template(template=prompt_str)
        else:
            raise Exception("Cannot load message templates")
