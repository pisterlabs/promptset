from langchain.agents import AgentType


OPENAI_API_KEY = "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"

ONLY_INDEX_MODE = False
OPENAI_llM_MODEL = "gpt-3.5-turbo"
AI_AGENT_TYPE = AgentType.ZERO_SHOT_REACT_DESCRIPTION
LLM_TEMPERATURE = 0.5
AI_AGENT_RETURN_DIRECT = False

LOCAL_CACHE = False
INDEX_HTML_PATH = "./templates/index.html"
RULES_kNOWLEDGE_PATH = "./documents/Rules.md"
AI_AGENT_TEMPLATE_PATH = "./rulegenerator/templates/rule_generator_template.txt"
HISTORY_PATH = "./history.txt"
COLLECTION_NAME = "rule-building-knowledge-base"