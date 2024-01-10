import unittest
from unittest.mock import MagicMock
from app.model import Message, StepData
from app.task_resolver.step_resolvers import GatherBookingInfoResolver
import os 
import openai
from dotenv import load_dotenv, find_dotenv

class TestGatherBookingInfoResolver(unittest.TestCase):
    
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
                ],
                "step_chat_history": [
                    Message("user", "Hola"),
                ],
                "expected_resolver_done": False
            },
            {
                "messages": [
                    Message("user", "Hola"),
                    Message("assistant", "Hola, ¿en qué puedo ayudarte?"),
                    Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene.")
                ],
                "step_chat_history": [
                    Message("user", "Hola"),
                    Message("assistant", "Hola, ¿en qué puedo ayudarte?"),
                    Message("user", "Me gustaría reservar una casa para dos personas, para el jueves que viene.")
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
                "expected_resolver_done": True
            }
        ]
        
        for idx, test in enumerate(test_cases):
            print(f"Running test {idx}")
            resolver = GatherBookingInfoResolver()

            # Act
            result = resolver.run(test["messages"], {}, test["step_chat_history"])
            
            # Assert
            self.assertIsNotNone(result)
            self.assertEqual(test["expected_resolver_done"], resolver.is_done())

    def test_run_exit_task_resolver_false(self):

        conversations = [
            [
                
            ]
        ]
        resolver = GatherBookingInfoResolver()
        # step_data = {"current_task_name": "MAKE_RESERVATION_TASK"}
        for idx, conv in enumerate(conversations):
            print(f"Running test {idx}")
            previous_steps_data = dict()
            
            # Act
            resolver.run(conv, previous_steps_data)

            # Assert
            self.assertEqual(resolver.is_done(), False)


if __name__ == '__main__':
    unittest.main()
