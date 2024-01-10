import os

import pandas as pd
import pytest
from anthropic import Anthropic

from springtime.services.excel_analyzer import ClaudeExcelAnalyzer
from springtime.services.sheet_processor import (
    CLAUDE_SHEET_PROCESSOR,
)
from springtime.services.table_analyzer import TableAnalyzer, TableAnalyzerImpl

XLSX = os.path.join(os.path.dirname(__file__), "../data/dummy-extracted.xlsx")


CLAUDE_EXCEL_ANALYZER = ClaudeExcelAnalyzer(Anthropic())


@pytest.fixture()
def claude_table_analyzer():
    return TableAnalyzerImpl(CLAUDE_EXCEL_ANALYZER, CLAUDE_SHEET_PROCESSOR)


def test_analyze(
    gpt_table_analyzer: TableAnalyzer,
    claude_table_analyzer: TableAnalyzer,
):
    xl = pd.ExcelFile(XLSX)
    for svc in (gpt_table_analyzer, claude_table_analyzer):
        resp = svc.analyze(
            excel_file=xl,
        )
        chunks = resp.chunks
        assert len(chunks) > 0
