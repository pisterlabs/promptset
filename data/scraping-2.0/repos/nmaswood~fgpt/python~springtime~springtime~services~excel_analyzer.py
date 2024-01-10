import abc

from anthropic import Anthropic
from pydantic import BaseModel

from springtime.services.format_sheet import format_sheet
from springtime.services.html import html_from_text
from springtime.services.prompts import CLAUDE_PROMPT
from springtime.services.sheet_processor import PreprocessedSheet


class ResponseWithPrompt(BaseModel):
    prompt: str
    content: str
    html: str | None


class ExcelAnalyzer(abc.ABC):
    @abc.abstractmethod
    def analyze(self, *, sheets: list[PreprocessedSheet]) -> ResponseWithPrompt:
        pass


class ClaudeExcelAnalyzer(ExcelAnalyzer):
    def __init__(self, anthropic_client: Anthropic) -> None:
        self.anthropic = anthropic_client

    def analyze(self, *, sheets: list[PreprocessedSheet]) -> ResponseWithPrompt:
        table_content = "\n---\n".join([format_sheet(sheet) for sheet in sheets])
        prompt = f"""


Human: {CLAUDE_PROMPT}

__START_DATA__
{table_content}
__END_DATA__


Assistant:
"""
        content = self.anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=1_000_000,
            prompt=prompt,
        ).completion.strip()

        return ResponseWithPrompt(
            prompt=prompt,
            content=content,
            html=html_from_text(content),
        )
