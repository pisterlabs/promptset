import re
import logging
from openai import OpenAI, AsyncOpenAI

from generators.entity_generator import EntityGenerator
from llm import LLM
from data_access_object import DataAccessObject


class AgentGenerator(LLM, EntityGenerator):
    def __init__(
            self, 
            gpt_client: OpenAI, 
            async_gpt_client: AsyncOpenAI, 
            dao: DataAccessObject) -> None:
        LLM.__init__(self, gpt_client, async_gpt_client)
        EntityGenerator.__init__(self)
        self.dao = dao

        self.system_prompt = self.dao.get_prompt_by_name('AG_SystemPrompt')

    def create(self):
        try:
            nationality = self.dao.get_random_nationality()
            # nationality = self.dao.get_nationality_by_id(1)
            occupation = self.dao.get_random_occupation()
            # occupation = self.dao.get_occupation_by_id(41)
            traits = self.dao.get_n_random_traits(5)
            investment_style = self.dao.get_random_investment_style()
            risk_tolerance = self.dao.get_random_risk_tolerance()
            communication_style = self.dao.get_random_communication_style()

            agent_name = self.generate_agent_attribute(
                'AG_GenAgentName', tok_limit=10, temp=1.25,
                 nationality=nationality, occupation=occupation
            )
            agent_handle = self.generate_agent_attribute(
                'AG_GenAgentHandle', tok_limit=10, temp=1.5, 
                traits=traits, communication_style=communication_style
            )
            agent_bio = self.generate_agent_attribute(
                'AG_GenAgentBio', tok_limit=150, temp=1.25,
                nationality=nationality, occupation=occupation, 
                agent_name=agent_name, traits=traits
            )

            agent_balance = self.generate_agent_attribute(
                'AG_GenAgentBalance', tok_limit=10, temp=1.3,
                agent_name=agent_name, agent_bio=agent_bio
            )
            agent_balance = self.convert_currency_to_decimal(agent_balance)

            agent = self.dao.insert(
                'agents',
                name=agent_name, handle=agent_handle, occupation=occupation,
                nationality=nationality, biography=agent_bio,
                investment_style=investment_style, risk_tolerance=risk_tolerance,
                communication_style=communication_style, balance=agent_balance
            )
            logging.info(f'Created new agent: {agent}')

            agent_goal = self.generate_agent_attribute(
                'AG_GenAgentGoal', tok_limit=150, temp=1.3,
                agent_name=agent_name, agent_bio=agent_bio
            )
            goal_embedding = self.generate_embedding(agent_goal)

            self.dao.insert(
                'memories', agent_id=agent.data[0]['id'],
                memory_details=agent_goal, embedding=goal_embedding
            )

            for trait in traits:
                self.dao.insert(
                    'agentstraits', agent_id=agent.data[0]['id'],
                    trait_id=trait['id'], is_positive=trait['is_positive']
                )

            return agent.data[0]['id']
        
        except Exception as e:
            # Properly handle exceptions and log the error
            logging.error(f'Failed to create new agent: {e}')
            raise

    def update(self, entity):
        pass

    def deactivate(self, agent):
        # Logic to deactivate an agent
        pass

    def generate_agent_attribute(self, prompt_name: str, tok_limit: int, 
                                 temp: float, **kwargs) -> str:
        if 'traits' in kwargs:
            traits_list = kwargs['traits']
            # Convert the list of trait dictionaries into a string representation
            traits_str = ', '.join([f"{trait['trait']}" 
                                    for trait in traits_list])

            kwargs['traits'] = traits_str

        prompt = self.dao.get_prompt_by_name(prompt_name).format(**kwargs)
        if prompt_name == 'AG_GenAgentBio':
            message = [{'role': 'system', 'content': self.system_prompt}, 
                       {'role': 'user', 'content': prompt}]
        else:
            message = [{'role': 'user', 'content': prompt}]

        logging.info(f'Prompt: {prompt}')

        agent_attribute = self.chat(message, temp=temp, max_tokens=tok_limit)
        logging.info(f'Generated attribute: {agent_attribute}')

        if not agent_attribute:
            raise ValueError(f'Failed to generate agent attribute with prompt: {prompt}')

        return agent_attribute
    
    @staticmethod
    def convert_currency_to_decimal(currency_str):
        # Remove non-numeric characters except the decimal point
        return re.sub(r'[^\d.]', '', currency_str)