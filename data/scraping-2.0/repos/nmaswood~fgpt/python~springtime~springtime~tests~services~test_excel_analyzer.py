import os

import pandas as pd
import pytest
from anthropic import Anthropic

from springtime.services.excel_analyzer import ClaudeExcelAnalyzer, ExcelAnalyzer
from springtime.services.sheet_processor import (
    CLAUDE_SHEET_PROCESSOR,
)

XLSX = os.path.join(
    os.path.dirname(__file__),
    "../data/wet-noses-sales-and-margin.xlsx",
)


@pytest.fixture()
def claude_analyzer():
    client = Anthropic()
    return ClaudeExcelAnalyzer(client)


@pytest.mark.skipif(False, reason="")
def test_analyze_claude(claude_analyzer: ExcelAnalyzer):
    xl = pd.ExcelFile(XLSX)
    sheets = CLAUDE_SHEET_PROCESSOR.preprocess(xl=xl)
    resp = claude_analyzer.analyze(sheets=sheets)
    breakpoint()
