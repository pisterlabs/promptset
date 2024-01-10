# /Tools/Plan.py

from instructor import OpenAISchema
from pydantic import Field
from Utilities.Log import Log, Debug, type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agents.BaseAgent import BaseAgent
    from Agency.Agency import Agency
    from Agency.Arsenal import SHARED_TOOLS

class Plan(OpenAISchema):
    """
    Generates a workflow of actionable steps.
    """

    goal: str = Field(
        ...,
        description="The goal that the agent would like to achieve. Will be used as the basis for planning"
    )
    context: str = Field(
        ..., 
        description="Additional considerations to be taken into account when planning"
    )

    def run(self, agency: 'Agency'):
            
        current_agent = agency.active_agent

        master_plan_creation = agency.plan is None

        prompt = "You are being engaged to create a plan. Review the following:\n\n"
        prompt += "User's Prompt: " + agency.prompt + "\n\n"
        prompt += f"Your goal is to: {self.goal}\n\n"
        prompt += f"Plan considerations:\n{self.context}\n\n"

        if master_plan_creation:
            # Add team details
            prompt += "# Team Composition: \n"
            for agent in agency.agents:
                agent:BaseAgent = agent
                prompt += f"## Name: \"{agent.name}\"\n"
                prompt += f"### Description: {agent.description}\n"
                # prompt += f"### Services: {agent.services}\n" # TODO: fix: empty array []
                prompt += "\n"
            prompt += "\n"
        else:
            prompt += f"# Agency's Plan\n\n{agency.plan}\n\n"
        
        # Add available tools to prompt:
        if master_plan_creation:
            toolkit = SHARED_TOOLS
        else:
            toolkit = current_agent.tools # todo i think the issue here is that its being fed the internal function
        
        prompt += "# Available Tools: \n"
        try:
            for tool in toolkit:
                schema = tool.openai_schema
                prompt += "## " + schema['name'] + "\n" 
                prompt += schema['description'] + "\n\n"
            prompt += "\n"
        except Exception as e:
            Log(type.ERROR, f"Error in Plan.py: {e}")
            Log(type.ERROR, f"Tools: {' | '.join([tool for tool in toolkit])}")
            Log(type.ERROR, f"master_plan_creation: {master_plan_creation}")
    
        # Instruction to review inputs and make a plan
        prompt += "# Plan Structure\n\n"
        prompt += "The plan is a workflow of **actionable steps** that will be executed to accomplish the mission.\n"
        prompt += "An actionable step is specific instruction conducted by a single agent via a tool usage\n"
        prompt += "The plan format adhere's to the following structure:\n"
        prompt += "<step_number> + \".\" + <agent_name> + \" using \" + <tool_name> + \": \" + <description of instruction or expected deliverable>\"\n"
        prompt += "\nExample of a simplified multi step workflow (for the user's prompt \"Create me a script\"):\n"
        prompt += "\t\"1. Coder using CreateFile: Create the script\"\n"
        prompt += "\t\"2. Coder using Delegate: Instruct QA to test the generated script, providing them instructions on how to execute\"\n"
        prompt += "\t\"3. QA using ExecutePyScript: Review execution results and provide appropriate feedback\"\n"
        prompt += "\t\"4. User Agent: Submit script back to user with execution instructions"
        prompt += "\tExample of a simple one liner plan (for the user's prompt \"I have a query\"):\n"
        prompt += "\t\"1. User Agent: I will respond to the user's prompt\"\n\n"
        
        # Plan tweaking
        prompt += "## Additional considerations:\n"
        prompt += "- Ensure the plan is manageable:\n"
        prompt += "  - Recognize and acknowledge if the mission is too complex.\n"
        prompt += "    - Size complexity will depend on the context so use your judgement.\n"
        prompt += "    - It is acceptable that the user's prompt is as simple as a one step plan\n"
        prompt += "  - Refuse plan generation when:\n"
        prompt += "    - The mission is too general and cannot be executed via actionable steps.\n"
        prompt += "    - The execution to achieve the desired result is deemed infeasible.\n"
        prompt += "    - The request falls outside the agent's capabilities.\n"
        prompt += "    - During refusals, provide detailed explanations:\n"
        prompt += "      - Why the mission cannot be carried out or the plan cannot be generated.\n"
        prompt += "      - Clarify what changes are needed for a successful attempt.\n"
        if master_plan_creation:
            prompt += "- Delegation is key:\n"
            prompt += "  - Each agent is equipped with 'Delegate' to perform the handoff of the tasks.\n"
            prompt += "  - The invocation of the tool 'Delegate' is to be it's own step in the plan, ensuring proper delegation.\n"

        prompt += "\n\n**THE GOAL IN PLAN CREATION IS TO SIMPLY CONSIDER THE MISSION AGAINST "
        if master_plan_creation:
            prompt += "THE ENVIRONMENT (AGENTS AND TOOLS AVAILABLE)"
        else:
            prompt += "YOUR CAPABILITIES"
        prompt += " WITH THE LEAST AMOUNT OF ACTIONABLE STEPS NECESSARY**\n\n"
        prompt += "Think step by step. Good luck, you are great at this!\n"
            
        Debug(f"Plan Prompt for {current_agent.name}:\n{prompt}")
        
        # todo: use SPR writer to compress this prompt statically (aka update this file to be more concise)
        
        if master_plan_creation:
            Log(type.ACTION, f"Agency is generating a plan\n")
        
        # todo: need to test whether its better to have the plan generated here,
        # or have this prompted returned as tool output for agent to decide what to do next
        plan = current_agent.get_completion(message=prompt, useTools=False)

        Log(type.RESULT, f"\nPlan Generated:\n{plan}\n")
        
        return plan
        
