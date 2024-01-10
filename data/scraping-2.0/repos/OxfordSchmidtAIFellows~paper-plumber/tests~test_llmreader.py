"""Tests for LLMReader class and utilities."""

import pytest
from paperplumber.parsing import OpenAIReader


@pytest.mark.parametrize(
    "target, text, expected",
    [
        (
            "enthalpy of formation",
            """The enthalpy of formation of water is 25.53 kcal/mol""",
            "25.53 kcal/mol",
        ),
        (
            "folding rate constant",
            """Figure 5A shows the refolding trajectory for Arc-L1-Arc following a jump from 7.0 to 2.4 M urea
            (3.1 µM protein, 25 °C, pH 7.5, 250 mM KCl, PMT voltage). The data fit well to a single exponential
            with a refolding rate constant (k f ) of 240 s -1 . Greater than 95% of the expected change in
            amplitude is observed during the data collection phase (i.e., less than 5% of the expected amplitude
            change occurs in the dead time). In experiments performed at final urea concentrations of 2.0 and
            2.8 M, k f was found to be independent of Arc-L1-Arc protein concentration from 1 to 10 µM (data
            not shown).""",
            "240 s^-1",
        ),
    ],
)
def test_openaireader(target, text, expected):
    assert OpenAIReader(target=target).read(text) == expected
