"""
File that contains the custome LangChain tool get_risk_scoring_system.
"""
import logging

from langchain.agents import tool

@tool
def get_risk_scoring_system(question: str) -> str: 
    """
    This is a useful tool for when you need a gudie on how to score risks (likelihood and impact).
    """
    logging.info("Risk scoring system forwarded")
    return """A scoring system for risk management helps quantify and assess risks based on their likelihood and impact. Here's a commonly used scoring system that combines likelihood and impact ratings:
    Likelihood Rating:
    1: Rare - The risk is unlikely to occur.
    2: Unlikely - The risk has a low probability of occurring.
    3: Possible - The risk may occur under certain circumstances.
    4: Likely - The risk is expected to occur.
    5: Almost Certain - The risk is highly probable.
    Impact Rating:
    1: Insignificant - Negligible impact on objectives or minimal consequences.
    2: Minor - Some impact on objectives, but manageable with minimal effort.
    3: Moderate - Significant impact on objectives, requiring additional resources and actions to address.
    4: Major - Severe impact on objectives, potentially causing delays, financial loss, or reputation damage.
    5: Catastrophic - Devastating impact on objectives, leading to significant losses or even business failure.
    Risk Scoring:
    Multiply the Likelihood Rating by the Impact Rating to calculate the risk score. For example, if a risk has a Likelihood Rating of 3 and an Impact Rating of 4, the risk score would be 3 x 4 = 12.
    Risk Rating:
    Based on the risk score, assign a Risk Rating to categorize the risk's severity:
    Low Risk: Risk score of 1-5
    Medium Risk: Risk score of 6-15
    High Risk: Risk score of 16 or above
    By using this scoring system, risks can be prioritized based on their potential impact, enabling risk managers to allocate appropriate resources and focus on addressing high-priority risks first. Keep in mind that this scoring system can be tailored to fit the specific needs and context of your organization or project."""