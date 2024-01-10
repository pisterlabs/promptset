from langchain import Runnable, Chain
from langchain.tools import Tool
from tokenizer import PreprocessingRunnerBranch
import logging

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level="INFO")

class AnalysisTool(Runnable):
    def __init__(self):
        self.preprocessing_runner_branch = PreprocessingRunnerBranch()
    
    def run(self, text, tokenizer_key):
        # Tokenization
        tokenized_text = self.preprocessing_runner_branch.tokenize(text, tokenizer_key)
        logger.info(f"Tokenized Text: {tokenized_text}")
        
        # Additional analysis, tagging, and processing can be added here
        ...
        return "Analysis completed."

# Define analysis_tool as a Tool
analysis_tool = Tool(
    name="AnalysisTool",
    func=AnalysisTool().run,
    description="A tool to analyze, tokenize, tag, and process unstructured data"
)

# Usage:
# Assuming AnalysisTool is at index 3 in the ALL_TOOLS list
analysis_result = tools[3].func("some unstructured text", 'bert')
