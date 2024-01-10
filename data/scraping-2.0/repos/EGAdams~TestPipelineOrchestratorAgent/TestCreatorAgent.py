from typing import List
from agency_swarm import set_openai_key
from agency_swarm import Agent
from tools import Codebase, RequirementsAnalysisTool, TestGenerationTool
from instructor import OpenAISchema
from typing import List
from strategies.CompileStrategy import CompileStrategy
from commands.TestPipelineCommand import TestPipelineCommand
from MessageSender import MessageSender

# import sys
# sys.path.append('/home/adamsl/')

from TestPipelineOrchestratorAgent import TestPipelineOrchestratorAgent


# set_openai_key(input("YOUR_API_KEY: "))
set_openai_key( "sk-PITEHd8tJ95HWyi8L3hmT3BlbkFJxBP3CE8iBm8BfwrBRMmw8" )

# Assuming Codebase, RequirementsAnalysisTool, TestGenerationTool are defined elsewhere
# and are compatible with the Instructor's BaseTool

class TestCreatorAgent( Agent ):
    def __init__( self, message_sender: MessageSender ):
        super().__init__(
            name="test_agent_local",
            files_folder="./files",
            instructions="./instructions.md",
            model="gpt-3.5-turbo-1106",
            tools=[ Codebase, RequirementsAnalysisTool, TestGenerationTool ])
        
        self.message_sender = message_sender


        self.test_orchestrator_agent_id = "TestPipelineOrchestratorAgent"

def request_test_generation( self ):
    try:
        # Assuming each tool has the relevant method as per their definition
        # code_analysis = self.codebase.run()
        # test_scenarios = self.requirements_tool.analyze_requirements()
        test_scenarios = []
        initial_failing_tests = self.test_generation_tool.generate_initial_failing_tests(
            test_scenarios)

        # Using message sender to communicate with TestPipelineOrchestratorAgent
        self.message_sender.send(
            to=self.test_orchestrator_agent_id,
            subject="Propose Tests",
            content={
                "initial_failing_tests": initial_failing_tests
            }
        )
    except Exception as e:
        # Handle exceptions appropriately
        print(f"An error occurred: {e}")
        # Additional error handling logic can be added here



# Example usage
if __name__ == "__main__":
    messageSender = MessageSender()
    developer = TestCreatorAgent( messageSender )
    # developer.request_test_generation()

    orchestrator = TestPipelineOrchestratorAgent()
    compile_command = TestPipelineCommand( CompileStrategy())
    orchestrator.set_command( compile_command )
    orchestrator.execute_command()
    # Further logic to add observers and execute different strategies
