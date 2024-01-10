"""Service to manage the LLM requests and responses."""
import os
import time
from nameko.extensions import DependencyProvider
from langchain import PromptTemplate

from nameko.rpc import rpc

from langchain import PromptTemplate
from langchain.llms import OpenAI


class LLMPromptFormatter:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def formatLLMPrompt(self, prompt: str, inputs: dict[str, str] = None):
        """Format the LLM prompt."""
        inputs = inputs or {}
        if not prompt:
            return ""
        prompt_template = PromptTemplate.from_template(prompt)
        return prompt_template.format(**inputs)

    def getLLMResponse(self, prompt: str):
        """Get the LLM response."""
        if not prompt:
            return ""
        llm = OpenAI(openai_api_key=self.api_key)
        return llm(prompt)


class LLMPromptProvider(DependencyProvider):
    """Dependency provider to manage the LLM request object."""

    redis_client = None
    open_ai_key = None

    def setup(self):
        """Setup the LLM request object."""
        # Get the OPENAI key
        self.open_ai_key = os.environ.get("OPENAI_KEY")
        print(f"Using OpenAI key: {self.open_ai_key}")

    def get_dependency(self, worker_ctx):
        """Return the LLM request object."""
        return self

    def sendPrompt(self, prompt_template: str, inputs: dict[str, str] = None):
        """Send the prompt to the LLM."""
        llm_formatter = LLMPromptFormatter(self.open_ai_key)
        prompt = llm_formatter.formatLLMPrompt(prompt_template, inputs)
        return llm_formatter.getLLMResponse(prompt)


class LLMService:
    name = "llm_service"

    llm_formatter = LLMPromptProvider()

    @rpc
    def get_llm_response(
        self, prompt_template: str, inputs: dict[str, str] = None
    ):
        """Run the LLM request and return the response."""
        return self.llm_formatter.sendPrompt(prompt_template, inputs)
