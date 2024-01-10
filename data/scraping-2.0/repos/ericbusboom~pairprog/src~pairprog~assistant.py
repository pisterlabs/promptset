import json
import logging
import unittest
import uuid
from copy import deepcopy
from datetime import datetime
from itertools import count

import openai
import tiktoken

from .objectstore import ObjectStore
from .tool import Done, PPTools, TaskState
from .util import *

logger = logging.getLogger(__name__)

# The specialized prompts for each state. The
# first is the auto continue prompt, which is Null when auto continue is off
# The second is the prompt is included in the system prompt
specializations = {
    TaskState.NONE: [None, ''],
    TaskState.ANALYZE: [
        '[ You are analyzing a task. Remember to call start_task(task_description="...") to start the task ]',
        get_prompt('analyze_task')
    ],
    TaskState.INTASK: [
        '[ You are working on a task:\n\n```\n{task_description}\n```\n\nRemember to call  done_with_task() when you are done ]',
        get_prompt('in_task')
    ],
    TaskState.AUTO_CONTINUE: ['Continue with task. If you are done with the task, call the done_with_task()', '']
}


class Assistant:
    tools: PPTools = None

    def __init__(
            self,
            messages: List[dict[str, Any]] = None,
            model="gpt-3.5-turbo-1106",
            token_limit: int = None,  # max number of tokens to submit in messages
    ):

        self.messages = messages or []
        self.responses = []
        self.chunks = []

        self.run_id = None

        self.models = None
        self.model, self.token_limit = self.select_model(model, token_limit)

        self.tokenizer = tiktoken.encoding_for_model(self.model)

        self.session_id = datetime.now().isoformat() + '-' + str(uuid.uuid4())

        self.client = openai.OpenAI()

        self.iter_key = lambda v: f"/none/{v}"

        self.task_state = TaskState.NONE
        self.task_description = ''

    def set_tools(self, tools: PPTools):
        self.tools = tools
        self.tools.set_assistant(self)

    @property
    def func_spec(self):
        if self.tools:
            return self.tools.specification()
        else:
            return None

    def select_model(self, model, token_limit):
        """Resolve sort model codes to model names, and return the model name and token limit"""

        models_json = Path(__file__).parent.joinpath('models.json').read_text()
        models = json.loads(models_json)

        self.models = {m['model_name']: m for m in models}

        if str(model) == '3.5':
            model = "gpt-3.5-turbo-1106"
        elif str(model) == '4':
            model = "gpt-4"
        elif model not in self.models:
            raise Exception(f"Unknown model: {model}")

        d = self.models[model]

        return d['model_name'], token_limit or (d['context_window'] - d['output_tokens'])

    def display(self, m):
        """Display a message to the user"""
        print(m)

    @property
    def last_content(self):
        """Return the last content"""

        return self.messages[-1]

    def request_messages(self, messages=None, max_tokens=None, elide_args=True):
        """Return a set of request messages that are less than the token limit, and
        perform some other editing, such as eliding the arguments form tool requests"""
        if max_tokens is None:
            max_tokens = self.token_limit

        rm = []

        for m in reversed(messages or self.messages):

            if 'tool_calls' in m and elide_args:
                m = deepcopy(m)
                for tc in m['tool_calls']:
                    pass
                    # tc['function']['arguments'] = '<arguments elided>'
                if 'function_call' in m:
                    del m['function_call']

            toks = len(self.tokenizer.encode(json.dumps(m)))

            if toks < max_tokens:
                max_tokens -= toks
                rm.insert(0, m)
            else:
                break

        # Remote gets cranky if you have a tool call response with
        # no tool calls, so remove it if it is there
        if rm and 'tool_call_id' in rm[0]:
            rm = rm[1:]

        specialization = specializations[self.task_state][1].format(
            task_description=self.task_description
        )

        sm = self.tools.system_message().format(specialization=specialization)

        logger.debug(log_debug(f"System message: {sm}"))

        return [{'role': 'system', 'content': sm}] + rm

    def stop(self):
        pass

    def print_content_cb(self, chunk):
        # clear_output(wait=True)

        content = chunk.choices[0].delta.content

        if content is not None:
            print(content, end='')

    def process_streamed_response(self, g, call_back=None):
        """Read the streamed responses from a call to the chat completions interface, and call the callback
        for each chunk. Returns all of the chunks aggregated into one chunk, and a list of each response """
        from copy import deepcopy

        chunk = None
        responses = []

        for r in g:

            if call_back is not None:
                call_back(r)

            responses.append(r)

            if chunk is None:
                chunk = deepcopy(r)
            else:
                # Copy over the content changes
                for j, (chunk_choice, r_choice) in enumerate(zip(chunk.choices, r.choices)):
                    if chunk_choice.delta.content is None:
                        chunk_choice.delta.content = r_choice.delta.content or ''
                    else:
                        chunk_choice.delta.content += r_choice.delta.content or ''

                    # Then the tool calls
                    if r_choice.delta.tool_calls:
                        if not chunk_choice.delta.tool_calls:
                            chunk_choice.delta.tool_calls = r_choice.delta.tool_calls
                        else:

                            for (chunk_tc, r_tc) in zip(chunk_choice.delta.tool_calls, r_choice.delta.tool_calls):
                                chunk_tc.function.arguments += r_tc.function.arguments

                    # Copy the finish reason. We are assigning this, not appending it, because we really only
                    # want thte last one
                    chunk_choice.finish_reason = r_choice.finish_reason

        return chunk, responses

    def _run(self, prompt: str | List[dict[str, Any]], streaming=True, **kwargs) -> str | None:
        """Run a  completion request loop"""

        if isinstance(prompt, str):
            self.messages.append({"role": "user", "content": prompt})
        else:
            self.messages.extend(prompt)

        finish_reason = None

        for iteration in count():

            self.iter_key = lambda v: f"/loop/{v}/{iteration:03d}"

            g = self.client.chat.completions.create(
                messages=self.request_messages(),
                tools=self.tools.specification(),
                model=self.model,
                stream=True,
                timeout=10
            )

            chunk, responses = self.process_streamed_response(g, call_back=self.print_content_cb)

            # Clean up the content response so that it is just the content, not any tool calls or other
            # stuff
            self.messages.append({'role': 'assistant', 'content': chunk.choices[0].delta.content})

            self.responses.append(responses)
            self.chunks.append(chunk)

            finish_reason = chunk.choices[0].finish_reason

            match finish_reason:
                case "stop" | "content_filter":
                    self.stop()
                    return finish_reason
                case "length":
                    self.stop()
                    return finish_reason
                case "function_call" | "tool_calls":
                    try:
                        messages = self.call_function(chunk)
                        self.messages.extend(messages)
                    except Done:
                        self.stop()
                        return finish_reason

                case "null":
                    pass  # IDK what to do here.

                case _:
                    logger.debug(f"Unknown finish reason: {finish_reason}")

        return finish_reason

    def run(self, line=None) -> Any:

        line = line.strip() if line else None
        line = line if line else None

        def getline():
            return input('$> ')

        match line, self.task_state:
            case None, TaskState.NONE:  # No line, not in task
                line = getline()
            case _, TaskState.NONE:  # Line, in task
                pass  # line provided in argument

            case None, TaskState.ANALYZE:
                line = getline()
                line += "\n" + specializations[TaskState.ANALYZE][0]
            case _, TaskState.ANALYZE:
                line += "\n" + specializations[TaskState.ANALYZE][0]

            case _, TaskState.INTASK:
                # We don't prompt the user for input when in task mode
                line = specializations[TaskState.INTASK][0].format(task_description=self.task_description)

            case _, TaskState.AUTO_CONTINUE:
                line = specializations[TaskState.AUTO_CONTINUE][0]

            case _, _:
                # This should never happen, but if it does, we need
                # to reset the task state
                self.task_state = TaskState.NONE
                logger.error(log_error(f"Unknown task state: {self.task_state}"))
                line = getline()

        logger.info(log_debug(self.task_state.name))
        logger.debug(log_debug(f"Line: {line}"))

        if isinstance(line, str) and line.startswith('%'):
            line = line[1:]
            if line == 'stop':
                raise Done
            elif line == 'messages':
                # Print messages array as json
                print(json.dumps(self.messages, indent=2))
                return ''
            elif line == 'request':
                # Print messages array as json
                print(json.dumps(self.request_messages(), indent=2))
                return ''
            elif line == 'system':
                # Print messages array as json
                print(self.tools.system_message())
                return ''
            elif line == 'spec':
                print(json.dumps(self.tools.specification(), indent=2))
                return ''
            elif line.startswith('task'):
                # Start task mode
                self.task_state = TaskState.ANALYZE
                line = line.replace('task', '').strip()
            else:
                print(f"Unknown command %{line}")
                return ''

        return self._run(line)

    def count_tokens(self, r):

        if not isinstance(r, str):
            r = json.dumps(r, indent=2)

        return len(self.tokenizer.encode(r))

    def call_function(self, chunk):
        """Call a function references in the response, the add the function result to the messages"""

        delta = chunk.choices[0].delta

        tool_request = delta.model_dump()
        tool_request['content'] = ''

        messages = []

        for tool_call in delta.tool_calls:

            m = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": None
            }

            self.display(log_system(f"Calling tool '{tool_call.function.name}'({tool_call.function.arguments[:500]})"))

            try:
                r = self.tools.run_tool(tool_call.function.name, tool_call.function.arguments)
            except Exception as e:

                e_msg = f"Failed to call tool '{tool_call.function.name}' with args '{tool_call.function.arguments}': {e} "

                logger.error(log_error(e_msg))
                m['content'] = str(e_msg)
                # messages.append(m)
                return []

            else:

                if not isinstance(r, str):
                    r = json.dumps(r, indent=2)

                toks = self.count_tokens(r)

                if toks > self.token_limit / 2:
                    max_preview = int(len(r) / toks * (self.token_limit / 4))

                    r = 'ERROR: Response too long to return to model. Try to make it smaller. ' + \
                        'Here is the first part of the response: \n\n' + r[:max_preview]

                    logger.error(log_error('Response too long to return to model.'))
                else:
                    logger.info(log_system(f"Response: {r[:200]}"))

                m['content'] = r
                messages.append(m)
        return [tool_request] + messages


class MyTestCase(unittest.TestCase):
    def test_basic(self):
        import typesense
        from pathlib import Path
        from pairprog.taskmachine import TaskManager, logger as tm_logger

        logging.basicConfig()
        tm_logger.setLevel(logging.INFO)

        # rc = ObjectStore.new(bucket='test', class_='FSObjectStore', path='/tmp/cache')

        rc = ObjectStore.new(name='barker_minio', bucket='agent')

        ts = typesense.Client(
            {
                "api_key": "xyz",
                "nodes": [{"host": "barker", "port": "8108", "protocol": "http"}],
                "connection_timeout_seconds": 1,
            }
        )

        tool = TaskManager(ts, rc.sub('task-manager'), Path('/Volumes/Cache/scratch'))

        assis = Assistant()
        assis.set_tools(tool)

        assis.run("what time will it be in one and one half hours? I am in the Pacific time zone")

        print("=" * 80)

        print(assis.request_messages())

    def test_codeand_store(self):
        import typesense
        from pathlib import Path

        logging.basicConfig()
        logger.setLevel(logging.DEBUG)

        rc = ObjectStore.new(name='barker_minio', bucket='agent')

        ts = typesense.Client(
            {
                "api_key": "xyz",
                "nodes": [{"host": "barker", "port": "8108", "protocol": "http"}],
                "connection_timeout_seconds": 1,
            }
        )

        tool = PPTools(ts, Path('/Volumes/Cache/scratch'))

        assis = Assistant(None)

        assis.run("Search your filesystem for information about eric")


if __name__ == "__main__":
    unittest.main()
