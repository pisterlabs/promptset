"""
Copyright (c) VKU.NewEnergy.

This source code is licensed under the Apache-2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Load langchain configs contains LLMChain model and Prompt.
- LLMChain: https://python.langchain.com/docs/modules/model_io/models/llms/llm_serialization
- Prompt: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/prompt_serialization
"""
import os.path
import yaml
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.prompts import load_prompt

from core.constants import BaseConstants

class LangChainDataLoader:
    """Langchain Data loader."""
    config: dict[str, dict[str, str]]
    prompts: dict[str, BasePromptTemplate]
    
    def __init__(self):
        self.prompts = {}

        with open(os.path.join(BaseConstants.ROOT_PATH, "configs/prompts/config.yaml"), "r") as f:
            self.config = yaml.safe_load(f)

        self._load_prompt()

    def _load_prompt(self):
        """Load prompt."""
        for title, info in self.config.items():
            self.prompts[title] = load_prompt(os.path.join(BaseConstants.ROOT_PATH, info["filepath"]))

    def preprocessing_qa_prompt(
        self,
        language: str,
        chat_history = None,
    ):
        prompt_title = "qaPrompt"
        
        qa_template = self.prompts[prompt_title].template
        qa_template += (
                    f"Based on the conversation history and the new question of customer, "
                    f"write a helpful response in {language} language"
                )
        
        qa_template += "\nResponse:\n\n"

        qa_template = qa_template.format(
                                context="{context}",
                                question="{question}",
                            )
        self.prompts[prompt_title] = PromptTemplate(template=qa_template, input_variables=["context", "question"])