from langchain.prompts import StringPromptTemplate
from pydantic.v1 import BaseModel, validator

PROMPT = """\
I want you to act as the author of a commit message in git.
I'll enter a git diff, and your job is to convert it into a useful commit message.
Do not preface the commit with anything, use the present tense, return the full sentence, and use the conventional commits specification (: ):
{diff}
"""


class CommitNameGeneratorPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes a code diff as input and formats the prompt template to generate a commit message from a git diff."""

    @validator("input_variables")
    def validate_input_variables(cls, variables) -> str:
        """Validate that the input variables are correct."""
        if len(variables) != 1 or "diff" not in variables:
            raise ValueError("diff must be the only input_variable.")
        return variables

    def format(self, **kwargs) -> str:
        prompt = PROMPT.format(diff=kwargs["diff"])
        return prompt

    def _prompt_type(self):
        return "commit-name-generator"
