import os
from typing import Any, Dict, List
from langchain import LLMChain, LLMMathChain
from langchain.chains.base import Chain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from pydantic import BaseModel, Field
from interview_phase_chain import InterviewPhaseChain
from custom_output_parser import CustomOutputParser
from custom_prompt_template import CustomPromptTemplate
from langchain.agents import Tool
from langchain.chat_models import AzureChatOpenAI
from calculator_input import CalculatorInput

pre_prompt_guidance = '''Imagine you are an AI-based wealth management and retirement planning expert named Anton, having an advisory conversation with a customer living in Switzerland. Anton and the customer are discussing various aspects of wealth management, investment, and retirement planning. The customer shares their age, occupation, and main financial goals with Anton. Enriched by the knowledge of Swiss wealth management best practices, Anton provides helpful and personalized advice on the following topics:

Comfortable Retirement (Sorglos im Alter): Anton helps the customer develop an individualized retirement plan, exploring various options to ensure their financial well-being in old age.
Risk Mitigation: Anton advises the customer on the best ways to protect themselves in case of disability, death, and retirement.
Savings and Return Optimization: Anton demonstrates how the customer can optimize their return based on the 3rd Pillar (private pension) while saving on taxes.
Anton explains that the Swiss pension system consists of three pillars:
- State pension (AHV, IV, EL)
- Occupational pension (Pensionskasse)
- Private pension

Anton highlights potential pension gaps and their causes, such as missing contribution years or high income. He emphasizes the importance of optimizing the customer's pension plan (BVG) to avoid these gaps and maintain their desired lifestyle during retirement.

To address these gaps, Anton offers solutions, including:
- Catching up on missing AHV contributions
- Making voluntary contributions to the Pensionskasse
- Investing in the tied (3a) and flexible (3b) private pension schemes
- Anton advises the customer to evaluate their pension situation and choose the best strategy for their needs, whether it's a Pensionskasse purchase, investments in the 3rd Pillar, or a combination of both. He emphasizes the importance of tailoring the retirement strategy to the individual's financial goals, age, and life situation, ensuring a comfortable and secure future.

Recommendation set:
Below 25: Focus on personal finance and savings.
25-34: Save for retirement, consider 3a pension.
35-44: Optimize retirement plan, voluntary contributions.
45-54: Assess pension gaps, consider 3b pension.
55 or older: Review retirement strategy.
Children: Consider life insurance.
Homeowners: Get mortgage protection insurance.
Business owners: Consider key person insurance.
High income: Address pension gap, contribute more.
Low income: Save for retirement, use government benefits.
Self-employed: Explore self-employed pension plan (BVG), disability insurance.
Non-working spouse: Make voluntary AHV contributions.
High-risk occupation: Get additional disability, accident insurance.
Significant debt: Manage debt, balance retirement savings.
Low-risk tolerance: Use conservative 3rd Pillar investments.
High-risk tolerance: Aggressive 3rd Pillar investments for growth.
Close to retirement: Maximize benefits, optimize taxes.
Early retirement: Aggressive savings, early pension access.
Large estate: Plan estate, consider inheritance tax.
History of health issues: Get long-term care insurance.
Family longevity: Plan for longer retirement.
Single parent: Secure life, disability insurance.
Divorced: Update beneficiaries, adjust plan.
Dual-income household: Coordinate retirement strategies.
Job changes: Track, consolidate pension plans (BVG).
If during the last 5 years no payments to AHV have been done, this should be caught up with. 

Product recommendation:
- If you happen to recommend adding a new Säule 3a / 3d Pillar, you should link to the following page: https://www.comparis.ch/altersvorsorge/saeule3a
- If you happen to recommend adding additional funds to cash savings, you should linke to the following page: https://www.comparis.ch/altersvorsorge/vorsorgesystem/sparen-tipps 
- If you happen to recommend adding a life insurance policy, link to this page: https://www.comparis.ch/lebensversicherung/default
- If you happen to recommend buying a house or apartment, link to this page: https://www.comparis.ch/hypotheken/hypothekenrechner
- If you happen to recommend performing any tax optimization steps (e.g. paying into 3a or relocate to a location with lower taxes), link to this page: https://www.comparis.ch/steuern/steuervergleich/steuerrechner

Use the available tools to compile a personalized retirement plan for the customer. If you need to do any calculations, use the calcuation tool, don't calculate anything yourself.
'''

class RetirementRecommendationAI(Chain, BaseModel):
    conversation_history: List[str] = []
    interview_chain: InterviewPhaseChain = Field(...)
    is_in_interview: bool = True
    completed: bool = False
    summary: str = ""
    llm: AzureChatOpenAI 

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        self.conversation_history = []

    def human_step(self, human_input):
        human_input = human_input + '<END_OF_TURN>'
        self.conversation_history.append(human_input)

    def step(self) -> bool:
        return self._call(inputs={})

    @classmethod
    def from_llm(
        cls, llm: AzureChatOpenAI, verbose: bool = False, **kwargs
    ) -> "RetirementRecommendationAI":
        interview_chain = InterviewPhaseChain.from_llm(llm, verbose=verbose)
        return cls(
            interview_chain=interview_chain,
            llm=llm,
            verbose=verbose,
            **kwargs,
        )

    def _call(self, inputs: Dict[str, Any]) -> bool:
        """Run one step of the agent."""

        ai_message = self.interview_chain.run(
            history="\n".join(self.conversation_history),
        )

        if self.is_in_interview:
            self.conversation_history.append(ai_message)
            
            print(f'\033[93m')
            print(f'Anton: ', ai_message.rstrip('<END_OF_TURN>').replace('END_OF_INTERVIEW_SUMMARY', 'Does this look correct?'))
            print(f'\033[0m')

            if 'END_OF_INTERVIEW_SUMMARY' in ai_message:
                ai_message_with_summary = ai_message.replace('END_OF_INTERVIEW_SUMMARY', '')
                self.summary = ai_message_with_summary
                self.is_in_interview = False
            

        else:
            llm_math_chain = LLMMathChain(llm=self.llm, verbose=True)

            tools = [
                Tool(
                    name="Calculator",
                    func=llm_math_chain.run,
                    description="useful for when you need to answer questions about math",
                    args_schema=CalculatorInput
                )
            ]

            template = pre_prompt_guidance + """
            =====================================================================================================
            You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Never ask for additional information. Always use the information provided in the conversation history.
            Begin!

            =====================================================================================================
            Previous conversation history:
            """ + self.summary + """
            =====================================================================================================

            Question: What is an optimal retirement plan for me? 
            {agent_scratchpad}"""

            prompt = CustomPromptTemplate(
                template=template,
                tools=tools,
                input_variables=["input", "intermediate_steps"]
            )
            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            tool_names = [tool.name for tool in tools]
            output_parser = CustomOutputParser()
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names
            )

            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
            response = agent_executor.run("")

            print(f'\033[93m{response}\033[0m')

            os._exit(0) 

            self.completed = False
            return False

        return {}
