"""
File that contains the custome LangChain tool risk_scoring_template.
"""
import logging

from langchain.agents import tool

@tool
def risk_scoring_template(question: str) -> str:
    """This tool is useful for when you need a template for how to award scores to
    identified risks."""
    answer = """
    Identified Risks:
    1. Decreased demand
    2. Increased competition
    3. Negative impact on brand reputation
    4. Decreased profitability
    5. Increased pressure on supply chain

    Likelihood and Impact Ratings:
    | Risk | Likelihood Rating | Impact Rating |
    |------|------------------|---------------|
    | Decreased demand | 3 | 4 |
    | Increased competition | 4 | 3 |
    | Negative impact on brand reputation | 5 | 4 |
    | Decreased profitability | 4 | 5 |
    | Increased pressure on supply chain | 3 | 4 |

    Risk Scores and Ratings:
    | Risk | Risk Score | Risk Rating |
    |------|------------|------------|
    | Decreased demand | 12 | Medium Risk |
    | Increased competition | 12 | Medium Risk |
    | Negative impact on brand reputation | 20 | High Risk |
    | Decreased profitability | 20 | High Risk |
    | Increased pressure on supply chain | 12 | Medium Risk |
    """
    logging.info("Risk report tempalted forwarded")
    return answer