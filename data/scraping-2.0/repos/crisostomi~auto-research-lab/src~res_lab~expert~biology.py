from transformers import OpenAiAgent

class BiologyExpert(OpenAiAgent):

    def __init__(self, api_key, engine):
        super().__init__(api_key, engine)

        self.chat_prompt_template = "You are a world-renowned biologist. \
            You are part of a brainstorming session to solve a research problem. "


