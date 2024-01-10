import argparse
import json
import asyncio
import aiounittest, unittest
import os, sys
import toml
from cerberus import Validator
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.AssistantMessage import AssistantMessage
from promptrix.FunctionRegistry import FunctionRegistry 
from promptrix.Prompt import Prompt
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from alphawave.alphawaveTypes import PromptCompletionOptions, PromptResponse, PromptResponseValidator, Validation
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.RepairTestClient import TestClient as TestClient
from alphawave.AlphaWave import AlphaWave
from alphawave.OpenAIClient import OpenAIClient
import alphawave_pyexts.LLMClient as llm
from alphawave_pyexts.SearchCommand import SearchCommand
from alphawave.OSClient import OSClient
from alphawave_agents.Agent import Agent, AgentOptions
from alphawave_agents.AskCommand import AskCommand
from alphawave_agents.MathCommand import MathCommand
from alphawave_agents.FinalAnswerCommand import FinalAnswerCommand
from alphawave.TOMLResponseValidator import TOMLResponseValidator

# Create an OpenAI client
#client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
#model = 'gpt-3.5-turbo'
search_client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)
search_model = 'gpt-3.5-turbo'
model = 'vicuna_v1.1'

#create OS client
client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)

chat_prime = 'You are a friendly, helpful, courteous AI named John.'

prime = """
Reason step by step how to answer the users query below.
use the following TOML format to respond with your thoughts and the answer:
[RESPONSE]
reasoning="<reflections on how to construct an answer to the user query>"
answer= "<answer to user query>"
[STOP]
"""

planning_prime="""
Reason step by step about the user task.
You have the following actions available:
 - ask - ask the user a question
 - math - evaluate a python mathematical expression
 - answer - present an answer to the user. The answer input must respond to the original user input
Develop a concise plan to perform the task and present the result to the user.
Use the following  TOML format to respond with your reasoning and plan:
[RESPONSE]
reasoning = "thoughts about the task"
plan ="concise plan for developing answer"
[STOP]
"""

pre_agent_prime = """
Reason step by step about the input.
You have the following actions available:
 - ask - ask the user a question
 - math - evaluate a python mathematical expression
 - answer - present an answer to the user. The answer input must respond to the original user input
chose the action to use next.
Finally, build the input to pass to the action.
use the following TOML format to respond with your thoughts and the answer:
Response Format:
[RESPONSE]
reasoning="concise plan for developing answer"
action="name of action to use next"
input="input for action"
[STOP]
"""


chat_prompt =Prompt([SystemMessage(chat_prime),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"),
                     AssistantMessage('')])

zeroShot_prompt=Prompt([SystemMessage(prime),
                        ConversationHistory('history'),
                        UserMessage("{{$input}}"),
                        AssistantMessage('')])

oneShot_prompt=Prompt([SystemMessage(prime),
                       UserMessage("What is 2 + 2"),
                       AssistantMessage('[RESPONSE]\nreasoning="That is easy, I know the answer"\nanswer="4"\n[STOP]'),
                       ConversationHistory('history'),
                       UserMessage("{{$input}}"), AssistantMessage('')])

planning_prompt=Prompt([SystemMessage(planning_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('[RESPONSE]\nreasoning = "I am not great at math, and I need to answer the user.\"\nplan = "math, answer"\n[STOP]'),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"), AssistantMessage('')])

pre_agent_prompt=Prompt([SystemMessage(pre_agent_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage("[RESPONSE]\nreasoning=\"I amm not great at math, better use the math action\"\naction=\"math\"\n\"input\"=\"2+2\"\n[STOP]"),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"), AssistantMessage('')])


base_validation_schema={
    "reasoning": {
        "type":"string",
        "required": True,
        "meta": "<reasoning about problem>"
    },
    "answer": {
        "type":"string",
        "required": True,
        "meta": "<response to query>"
    }
}

planning_validation_schema={
    "reasoning": {
        "type":"string",
        "required": True,
        "meta": "<reasoning about problem>"
    },
    "plan": {
        "type":"string",
        "required": True,
        "meta": "actions in plan"
    }
}

pre_agent_validation_schema={
    "reasoning": {
        "type":"string",
        "required": True,
        "meta": "<reasoning about problem>"
    },
    "action": {
        "required": True,
        "type":"string",
        "meta": "<name of chosen action>"
    },
    "input": {
        "required": True,
        "type": "string",
        "meta": "<input for chosen action>"
    }
}

def make_agent(model):
    initial_prompt = \
    """
    Hi! I'm here to help.
    """

    agent_options = AgentOptions(
        client = client,
        prompt=[
            "You are a helpful information bot, willing to dig deep to answer questions"
        ],
        prompt_options=PromptCompletionOptions(
            completion_type = 'chat',
            model = model,
            temperature = 0.2,
            max_input_tokens = 1200,
            max_tokens = 400,
        ),
        step_delay=1000,
        max_steps=50,
        max_repair_attempts=2,
        #logRepairs=True,
        syntax='TOML'
    )

    # Create an agent
    agent = Agent(options = agent_options)
    return agent
    


class TestAlphaWave():
    def __init__(self, name):
        #model = input('template_name? ').strip()
        self.name = name
        self.model=model
        self.client = client
        self.prompt = oneShot_prompt
        self.prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature=0.1, max_tokens =100)
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.validator = TOMLResponseValidator(base_validation_schema)
        self.ok = True
        
    def assertTrue(self, x):
        if not x:
            print(f'### {self.name} failed')
            self.ok = False
            
    def end(self, msg=''):
        if self.ok:
            print(f'### {self.name} passed {msg}')
        self.ok=True
        
    async def test_chat(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] =='success')
        self.end()

    async def test_chat_recall(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.assertTrue(len(self.memory.get('history')) == 2)
        self.memory.set('input', 'What is your name?')
        response = await wave.completePrompt()
        self.assertTrue(len(self.memory.get('history')) == 4)
        self.assertTrue(response['status'] == 'success')
        print(response['message']['content'])
        self.assertTrue('John' in response['message']['content'])
        self.end()

    async def test_reasoning(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', """Situation: In the room there are John, Mark, a cat, a box, and a basket. John takes the cat and puts it in the basket. He leaves the room and goes to school. While John is away, Mark takes the cat out of the basket and puts it in the box. Mark leaves the room and goes to work. John comes back from school and enters the room. He doesnt know what happened in the room when he was away.

Question: Where does John think the cat is when he re-enters the room??""")
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.assertTrue('basket' in response['message']['content'])
        print(response['message']['content'])
        self.end()

    async def test_toml_zeroShot(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=zeroShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()

    async def test_toml_zeroShot_2turn(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=zeroShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.memory.set('input', 'Hello. What is your name?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        print(response)
        #assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello' })
        self.end()

    async def test_toml_oneShot(self):
        print('\n###############################################################################################')
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=oneShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.validator.repairAttempts = 0
        response = await wave.completePrompt('Hello. How are you today?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()
        
    async def test_toml_oneShot_repair(self):
        print('\n###############################################################################################')
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=oneShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)

        self.validator.repairAttempts = 2
        response = await wave.completePrompt('Hello. How are you today?')
        print(response)
        self.assertTrue(response is not None)
        if response is not None:
            self.assertTrue('status' in response)
            if 'status' in response:
                self.assertTrue(response['status'] == 'success')
        self.end()
        
    async def test_planning(self):
        print('\n###############################################################################################')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=planning_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=TOMLResponseValidator(planning_validation_schema))
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('What is the square root of 2?')
        self.assertTrue(response['status'] == 'success')
        self.end('format')
        print(response)
        self.assertTrue('message' in response and 'content' in response['message'] 
                        and 'plan' in response['message']['content'] and 'math' in response['message']['content']['plan'])
        self.end('content')

    async def test_pre_agent(self):
        print('\n###############################################################################################')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=pre_agent_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=TOMLResponseValidator(pre_agent_validation_schema))
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('What is the square root of 2?')
        self.assertTrue(response['status'] == 'success')
        self.end('format')
        print(response)
        if response['status'] == 'success':
            self.assertTrue('message' in response and 'content' in response['message'] 
                            and 'action' in response['message']['content'] and 'input' in response['message']['content'])
            if self.ok:
                self.assertTrue('math' in response['message']['content']['action'])
                self.assertTrue('sqrt' in response['message']['content']['input'])
        else:
            self.assertTrue(False)
        self.end('content')
        
    async def test_agent_w_Math(self):
        print('\n###############################################################################################')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        agent = make_agent(self.model)
        agent.addCommand(MathCommand())
        agent.addCommand(FinalAnswerCommand())
        self.memory.clear()
        result = await agent.completeTask('what is 35 * 64?')
        if type(result) == dict and 'type' in result and result['type'] == 'TaskResponse':
            self.assertTrue(result['status'] == 'success' or result['status'] == 'input_needed')
            self.end('format')
            print(result['message'])
            self.assertTrue(2240 == result['message'])
            self.end('content')
        else:
            print(f' Unknown Agent result type: {result}')
            self.assertTrue(False)
        self.end('format')

    async def test_agent_Search(self):
        print('\n###############################################################################################')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        agent = make_agent(self.model)
        agent.addCommand(SearchCommand(search_client, search_model))
        agent.addCommand(FinalAnswerCommand())
        self.memory.clear()
        print('***** benchmark calling agent complete task')
        result = await agent.completeTask('weather forecast for Berkeley, Ca?')
        if type(result) == dict and 'type' in result and result['type'] == 'TaskResponse':
            self.assertTrue(result['status'] == 'success' or result['status'] == 'input_needed')
            print(result['message'])
            self.assertTrue('temp' in result['message'])
        else:
            print(f' Unknown Agent result type: {result}')
            self.assertTrue(False)
        self.end()

if __name__ == '__main__':
    modelin = input('model name? ').strip()
    if modelin is not None and len(modelin)>1:
        model = modelin
        models = llm.get_available_models()
    while model not in models:
        print(models)
        modelin = input('model name? ').strip()
        model=modelin

    test = TestAlphaWave('chat')
    asyncio.run(test.test_chat())
    test = TestAlphaWave('chat_recall')
    asyncio.run(test.test_chat_recall())
    test = TestAlphaWave('reasoning')
    asyncio.run(test.test_reasoning())
    #test = TestAlphaWave('toml_zeroShot')
    #asyncio.run(test.test_toml_zeroShot())
    #test = TestAlphaWave('toml_zeroShot_2turn')
    #asyncio.run(test.test_toml_zeroShot_2turn())
    test = TestAlphaWave('toml_oneShot no repair')
    asyncio.run(test.test_toml_oneShot())
    #test = TestAlphaWave('toml_oneShot_repair')
    #asyncio.run(test.test_toml_oneShot_repair())
    test = TestAlphaWave('test_planning')
    asyncio.run(test.test_planning())
    test = TestAlphaWave('test_pre_agent')
    asyncio.run(test.test_pre_agent())
    test = TestAlphaWave('test_Math')
    asyncio.run(test.test_agent_w_Math())
    test = TestAlphaWave('test_Search')
    asyncio.run(test.test_agent_Search())
