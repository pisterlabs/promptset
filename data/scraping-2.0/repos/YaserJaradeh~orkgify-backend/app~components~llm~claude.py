from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic

from app.components.llm.base import LLMStrategy
from app.core.config import settings


class Claude21Strategy(LLMStrategy):
    MODEL: str = "claude-2.1"
    client: Anthropic = None

    def __init__(self):
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.keywords = ["temperature", "max_tokens_to_sample"]

    def execute(self, **kwargs) -> str:
        if any([not kwargs.get(key) for key in ["system_prompt", "user_prompt"]]):
            raise ValueError("Missing required arguments")

        # Extract the required arguments
        system_prompt = kwargs.pop("system_prompt")
        user_prompt = kwargs.pop("user_prompt")

        if settings.VERBOSE:
            print(
                f"Executing strategy with system_prompt={system_prompt} and user_prompt={user_prompt}"
            )

        # remove arguments in kwargs that are not in the keywords list
        kwargs = {key: value for key, value in kwargs.items() if key in self.keywords}

        # Make the actual request to the OpenAI API
        response = self.client.completions.create(
            model=self.MODEL,
            prompt=f"{system_prompt}{HUMAN_PROMPT} {user_prompt}{AI_PROMPT}",
            **kwargs,
        )

        # Return the response
        return response.completion
