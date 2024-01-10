from prompt import PromptProvider
from utils import openai_completion
from logger import DevOpsAILogger
import json, yaml

with open("config.yaml", "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

FINAL_STR = ""
NEXT_STEP_STR = ""
ERROR_TEXT = ""
if config["language"] == "en":
    FINAL_STR = config["en_labels"]["final_str"]
    NEXT_STEP_STR = config["en_labels"]["next_step_str"]
    ERROR_TEXT = config["en_labels"]["error_text"]
else:
    FINAL_STR = config["cn_labels"]["final_str"]
    NEXT_STEP_STR = config["cn_labels"]["next_step_str"]
    ERROR_TEXT = config["cn_labels"]["error_text"]

class QaEngine:
    prompt_provider: PromptProvider = PromptProvider(original_problem="")
    logger : DevOpsAILogger
    def __init__(self):
        self.logger = DevOpsAILogger.get_instance(name='qaengine', config=config)
        super().__init__()

    def get_prompt_provider(self):
        return self.prompt_provider

    def parse_return_from_openai(self, rtn_text):
        parsed_text = ""
        found = False
        for line in rtn_text.split("\n"):
            if line.startswith(NEXT_STEP_STR) or line.startswith(FINAL_STR) > 0:
                found = True
            if found:
                parsed_text += line + "\n"
        return parsed_text
        
    def get_answer(self, question: str, history: str) -> str:
        messages = self.prompt_provider.get_messages(question)
        lastStep2Str = self.prompt_provider.lastStep2Str()
        self.logger.info(f"last step before call openai: \n {lastStep2Str}")
        rtn_items = openai_completion(messages, 1)
        parsed_text = self.parse_return_from_openai(rtn_items[0])
        if parsed_text.strip().startswith(FINAL_STR):
            self.prompt_provider.clean_steps()
        else:
            if parsed_text.strip() != "":
                self.prompt_provider.add_step_from_str(parsed_text)
            else:
                parsed_text = ERROR_TEXT
        lastStep2Str = self.prompt_provider.lastStep2Str()
        self.logger.info(f"last step after call openai: \n {lastStep2Str}")
        parsed_text = parsed_text.replace("\n", "<p>")
        return parsed_text
