from src.executor import Executor
from src.openai_model import OpenAIModel
from src.config import Config
from src.memory import Memory


# class TestPromptGenerator:
#     config = Config("tests/config_files/default.json")
#     memory = Memory(config)
#     llm = OpenAIModel(config)

#     def test_generate(self):
#         llm_response = PromptGenerator.generate(
#             self.llm,
#             self.config.app_description,
#             self.memory.states,
#             self.memory.error_states,
#         )

#         assert llm_response
