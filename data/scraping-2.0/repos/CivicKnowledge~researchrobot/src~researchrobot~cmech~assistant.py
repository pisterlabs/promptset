import json
from typing import Optional, List
from alive_progress import alive_bar
import openai
from pydantic import BaseModel, Field
from researchrobot.objectstore import ObjectStore,resolve_cache
from researchrobot.cmech.util import generate_tools_specification
from researchrobot.openai_tools import costs
import tiktoken

class SocMatch(BaseModel):
    soc_code: str = Field(..., description="The SOC code of the experience.")
    rationale: str = Field(..., description="Reason for assigning this SOC code")
    quality: float = Field(..., description="The quality of the match, where 1 is perfect and 0 is no match.")


class Experience(BaseModel):
    """Experience hold the title and description for a single job experience,
    and the SOC codes that match it."""

    title: str = Field(..., description="The title of the experience.")
    description: str = Field(..., description="The description of the experience.")

    soc_codes: Optional[List[SocMatch]] = Field(description="The SOC codes of the experience.",
                                                default=None)


class Profile(BaseModel):
    pass


class Functions:

    def __init__(self):
        from researchrobot import ObjectStore
        import typesense
        from researchrobot.cmech.classify import SocEdData, SocEdClassifier

        class_rc = ObjectStore.new(name='barker_minio', bucket='linkedin', prefix='ai_job_classif/v1')
        embed_rc =  ObjectStore.new(name='barker_minio', bucket='linkedin', prefix='cache/embed')

        self.client = typesense.Client({
            'api_key': 'xyz',
            'nodes': [{
                'host': 'barker',
                'port': '8108',
                'protocol': 'http'
            }],
            'connection_timeout_seconds': 1
        })

        self.data = SocEdData(data_cache=class_rc, embed_cache=embed_rc)
        self.sec = SocEdClassifier(self.data, self.client)

    def search_job(self, title: str, description: str) -> object:
        """Search for a job experience by title and description.

        This function searches for a job experience based on a given title and description.
        It returns an Experience object that matches the search criteria.

        Args:
            title (str): The title of the job experience to search for. This should be a
                         string representing the job title, such as 'Software Engineer' or 'Data Analyst'.
            description (str): The description of the job experience to search for. This should
                               be a detailed string describing the job role, responsibilities, or
                               any specific details relevant to the search.

        Returns:
            matches: a table, in dict "records" orientation,  of matching job experiences, sorted by relevance.
        """

        r = self.sec.job_search(title, description).head(10)
        return r.to_dict(orient='records')

    @classmethod
    def tools(cls):
        return generate_tools_specification(Functions)

class CMechAssistant:

    def __init__(self,
                 tools: List[dict],
                 messages: List[dict] = None,
                 model = 'gpt-3.5-turbo-1106',
                 cache: Optional[dict|ObjectStore] = None,
                 token_limit: int = 6000 # max number of tokens to submit in messages
                 ):

        self.tools = tools #  Functions to call, and function definitions
        self.func_spec = tools.tools()

        self.messages = messages or []

        self.cache = resolve_cache(cache)
        self.messages_cache = self.cache.sub("messages")

        self.run_id = None

        self.responses = []
        self.model = model
        self.token_limit = token_limit
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    @property
    def cost(self):
        """Return the cost of all stored responses"""
        cost = 0
        for r in self.responses:
            sch = costs[r.model]
            cost += r.usage.prompt_tokens*sch[0]/1000 + r.usage.completion_tokens/1000*sch[1]

        return cost


    @property
    def limited_messages(self):
        """return the tail of the messages array for a number of messages that
        is less than the token limit"""

        total = 0
        msg = []
        for m in reversed(self.messages):
            toks = len(self.tokenizer.encode(m['content'] or ''))

            if toks + total <= self.token_limit:
                total += toks
                msg.insert(0, m)
            else:
                break

        return msg

    def stop(self):
        pass

    def run(self, prompt: str|List[dict], **kwargs):
        """Run a single completion request"""
        import uuid
        from time import time

        client = openai.OpenAI()

        self.run_id = hex(int(time()))[2:] + "-" +  str(uuid.uuid4())

        if isinstance(prompt, str):
            self.messages.append([{"role": "user", "content": prompt}])
        else:
            self.messages.extend(prompt)

        with alive_bar() as bar:
            while True:

                try:

                    self.messages_cache['request/'+self.run_id] = self.messages

                    r = client.chat.completions.create(messages=self.limited_messages,
                                                   tools=self.func_spec,
                                                   model=self.model)

                    self.responses.append(r)

                    bar.text(f"{len(self.messages)} messages")
                    bar()

                    message = r.choices[0].message.model_dump()

                    # GPT errs if you return a null function call when there is a tool call
                    # I think it's all tools now, and function_call is deprecated?
                    if 'function_call' in message and message['function_call'] is None:
                        del message['function_call']

                    self.messages.append(message)

                    self.messages_cache['response/'+self.run_id] = self.messages

                except Exception as e:
                    print("!!!!!! EXCEPTION !!!!!!!")
                    print("RUN ID", self.run_id)
                    print(e)
                    print("!!!!!! RESPONSE !!!!!!!")
                    print(json.dumps(r.model_dump(), indent=2))
                    print("!!!!!! MESSAGES !!!!!!!")
                    print(json.dumps(self.messages, indent=2))

                    raise


                match r.choices[0].finish_reason:
                    case "stop"|"content_filter":
                        self.stop()
                        return
                    case "length":
                        self.stop()
                        return
                    case "function_call" |"tool_calls":
                        self.call_function(r)
                    case "null":
                        pass # IDK what to do here.



    def call_function(self, response):
        """Call a function references in the response, the add the function result to the messages"""

        tool_calls = response.choices[0].message.tool_calls

        for tool_call in tool_calls:

            f = getattr(self.tools, tool_call.function.name)
            args = json.loads(tool_call.function.arguments)
            r = f(**args)

            m = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps(r)
            }

            self.messages.append(m)

import unittest
class MyTestCase(unittest.TestCase):
    def test_something(self):
        import json
        from researchrobot.cmech.util import pretty_print_conversation

        ex = Experience(title="Software Engineer", description="I am a software engineer.")

        messages = [
            {'role': 'system', 'content': 'We are testing the function calling features of the API.'},
            {'role': 'user',
             'content': "I'm going to give you an Experience object, and I want you to call the search_job function with it."},
            {'role': 'user', 'content': json.dumps(ex.model_dump())}
        ]

        cache_config = dict(name='barker_minio', bucket='linkedin', prefix='assistant')

        cm = CMechAssistant(tools=Functions(), cache=cache_config)

        cm.run(messages)

        pretty_print_conversation(cm.messages)




if __name__ == '__main__':
    unittest.main()
