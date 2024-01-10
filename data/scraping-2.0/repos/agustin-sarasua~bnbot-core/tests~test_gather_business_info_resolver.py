import unittest
from unittest.mock import MagicMock
from app.model import Message, StepData
from app.task_resolver.step_resolvers import GatherBusinessInfoResolver
import os 
import openai
from dotenv import load_dotenv, find_dotenv

class TestGatherBusinessInfoResolver(unittest.TestCase):
    
    def setUp(self):
        _ = load_dotenv(find_dotenv(filename="../.env")) # read local .env file
        openai.api_key = os.environ['OPENAI_API_KEY']
    
    def test_run(self):
        # Arrange
        def _create_step_data(info) -> StepData:
            step_data = StepData()
            step_data.resolver_data =info
            return step_data
        
        test_cases = [
            {
                "messages": [
                    Message("user", "Hola"),
                    Message("assistant", "Hola, ¿en qué puedo ayudarte?"),
                    Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene por dos noches.")
                ],
                "step_chat_history": [
                    Message("user", "Hola"),
                    Message("assistant", "Hola, ¿en qué puedo ayudarte?"),
                    Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene por dos noches.")
                ],
                "expected_resolver_done": False
            },
            {
                "messages": [
                    Message("user", "Hola"),
                    Message("assistant", "Hola, ¿en qué puedo ayudarte?"),
                    Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene por dos noches.")
                ],
                "step_chat_history": [
                    Message("user", "Hola"),
                    Message("assistant", "Hola, ¿en qué puedo ayudarte?"),
                    Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene por dos noches.")
                ],
                "expected_resolver_done": False
            },
            # {
            #     "messages": [
            #         Message("user", "Hola"),
            #         Message("assistant", "Hola, ¿en qué puedo ayudarte?"),
            #         Message("user", "Me gustaría reservar @casa.en.altos para dos personas, para el jueves que viene por dos noches.")
            #     ],
            #     "step_chat_history": [
            #         Message("user", "Hola"),
            #         Message("assistant", "Hola, ¿en qué puedo ayudarte?"),
            #         Message("user", "Me gustaría reservar @casa.en.altos para dos personas, para el jueves que viene por dos noches.")
            #     ],
            #     "expected_resolver_done": True
            # }
        ]
        
        for idx, test in enumerate(test_cases):
            print(f"Running test {idx}")
            resolver = GatherBusinessInfoResolver()

            # Act
            result = resolver.run(test["messages"], {}, test["step_chat_history"])
            
            # Assert
            self.assertIsNotNone(result)
            self.assertEqual(test["expected_resolver_done"], resolver.is_done())


if __name__ == '__main__':
    unittest.main()
