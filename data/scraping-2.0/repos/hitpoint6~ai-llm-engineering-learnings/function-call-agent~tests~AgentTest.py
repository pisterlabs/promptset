from FuncCallAgent import Agent
import unittest
import openai
from getpass import getpass

openai.api_key =  getpass("Enter your openai api key: ")

def circumference_calculator(radius: float, something: float = 4.4) -> float:
    return 2 * 3.14 * radius

class AgentTest(unittest.TestCase):
    def test_circumference_calculator(self):
        agent = Agent.Agent([circumference_calculator], "gpt-4")
        desc = agent.create_function_description_(circumference_calculator)
        self.assertEqual(desc["name"], "circumference_calculator")
        parameters = desc["parameters"]
        self.assertEqual(parameters["properties"]["radius"]["type"], "number")
        self.assertEqual(parameters["properties"]["something"]["type"], "number")
        self.assertEqual(parameters["required"], ["radius"])

if __name__ == '__main__':
    unittest.main()