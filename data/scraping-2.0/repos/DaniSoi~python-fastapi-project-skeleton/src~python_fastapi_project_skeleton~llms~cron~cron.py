from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate


class CronExpressionGenerator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

        self.prompt = PromptTemplate(
            input_variables=["cron_prompt"],
            template=r"""
Below is text describing a cron expression.
Your goal is to:
- Convert the text to a valid cron expression.
- The cron expression you generate must match this regular expression: "^((\*|[0-9]|[1-5][0-9]|60) |(\*|[0-9]|[1-5][0-9]|60) |(\*|[0-9]|[1-2][0-9]|3[0-1]) |(\*|[0-9]|[1-9]|[1-2][0-9]|3[0-1]|4[0-6]|5[0-3]) |(\*|[0-9]|[1-9]|1[0-2]))(\*|\/[0-9]|[0-9\-,\/]+) (\*|\/[0-9]|[0-9\-,\/]+) (\*|\/[0-9]|[0-9\-,\/]+) (\*|\/[0-9]|[0-9\-,\/]+) (\*|\/[0-9]|[0-9\-,\/]+)$"
- Return only the generated cron expression and nothing else. The cron expression must be trimmed of all whitespace.
- If you cannot generate a valid cron expression, return an empty string.
Here are some examples:
- Text: A cron that runs every hour
- Cron: 0 * * * *

- Text: A cron that runs every 12 hours
- Cron: 0 */12 * * *
Below is the text:
- Text: A cron that runs {cron_prompt}
- Cron: [YOUR RESPONSE HERE]
            """,  # noqa: E501
        )

    async def generate_cron_expr(self, cron_prompt: str) -> str:
        result: str = await self.llm.apredict(
            self.prompt.format(cron_prompt=cron_prompt)
        )
        return result
