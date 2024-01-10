import httpx
import openai
import json
import time
import logging
import copy

from .prompt_builder import FunctionalPrompt, PromptMessage, Assistant, System, FunctionCall, FunctionResult
from .get_wikipedia import WikipediaApi

from .react_prompt import FunctionalReactPrompt, NewFunctionalReactPrompt, TextReactPrompt
from .prompt_templates import NoExamplesReactPrompt
from .toolbox import ToolBox, WikipediaSearch

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)


class LLMReactor:
    def __init__(self, model: str, toolbox: ToolBox, prompt: FunctionalPrompt,
                 reflection_generator, max_llm_calls: int, client):
        self.model = model
        self.toolbox = toolbox
        self.prompt = prompt
        self.reflection_generator = reflection_generator
        self.max_llm_calls = max_llm_calls
        self.client = client
        self.step = 0
        self.finished = False
        self.answer = None

    def openai_query(self, **args):
        if isinstance(self.prompt, FunctionalPrompt):
            # todo - we might optimize and only send the functions that are relevant
            # in particular not send the functions if function_call = 'none'
            args["tools"] = self.toolbox.tools
            if not "tool_choice" in args:
                args["tool_choice"] = "auto"
        else:
            args["stop"] = ["\nObservation:"]

        response = None
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.prompt.to_messages(),
            **args
        )
        response_message = completion.choices[0].message
        return response_message

    def set_finished(self):
        self.finished = True

    def process_prompt(self):
        logger.debug(f"Processing prompt: {self.prompt}")
        self.step += 1
        if self.step == self.max_llm_calls:
            response = self.openai_query(tool_choice={'type': 'function', 'function': {'name': 'finish'}})
        else:
            response = self.openai_query()
        function_call = self.prompt.function_call_from_response(response)
        if function_call:
            result = self.toolbox.process(function_call)
            message = FunctionCall(result.tool_name, **result.tool_args)
            logger.info(str(message))
            self.prompt.push(message)
            if result.tool_name == 'finish':
                self.answer = result.observations
                self.set_finished()
                return
            elif self.step == self.max_llm_calls:
                self.set_finished()
                logger.info("<<< Max LLM calls reached without finishing")
                return

            message = FunctionResult(result.tool_name, result.observations)
            logger.info(str(message))
            self.prompt.push(message)

            message = self.reflection_generator.generate(self.step, self.max_llm_calls)
            logger.info(str(message))
            self.prompt.push(message)
            response = self.openai_query(tool_choice='none')
            message = Assistant(response.content)
            logger.info(str(message))
            self.prompt.push(message)
        else:
            message = Assistant(response.content)
            logger.info(str(message))
            self.prompt.push(message)


def get_answer(question, config):
    print("\n\n<<< Question:", question)
    wiki_api = WikipediaApi(max_retries=2, chunk_size=config['chunk_size'])
    toolbox = WikipediaSearch(wiki_api)
    client = openai.OpenAI(timeout=httpx.Timeout(20.0, read=10.0, write=15.0, connect=4.0))
    reactor = LLMReactor(config['model'], toolbox, config['prompt'], config['reflection_generator'], config['max_llm_calls'], client=client)
    while True:
        print()
        print(f">>>LLM call number: {reactor.step}")
        reactor.process_prompt()
        # print(prompt.parts[-2])
        if reactor.finished:
            return reactor
#        if 'gpt-4' in config['model']:
#            time.sleep(59)
