#%%
from langchain.document_loaders import JSONLoader
import json
from pprint import pprint
from langchain.agents import create_json_agent, AgentExecutor
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec
os.environ["OPENAI_API_KEY"]='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'



#%%
json_string = {'image': {'creation_timestamp': 1675549016, 'uri': 'image_of_the_chat.jpg'},
     'is_still_participant': True,
     'joinable_mode': {'link': '', 'mode': 1},
     'magic_words': [],
     'messages': [{'content': 'Bye!',
                   'sender_name': 'User 2',
                   'timestamp_ms': 1675597571851},
                  {'content': 'Oh no worries! Bye',
                   'sender_name': 'User 1',
                   'timestamp_ms': 1675597435669},
                  {'content': 'No Im sorry it was my mistake, the blue one is not '
                              'for sale',
                   'sender_name': 'User 2',
                   'timestamp_ms': 1675596277579},
                  {'content': 'I thought you were selling the blue one!',
                   'sender_name': 'User 1',
                   'timestamp_ms': 1675595140251},
                  {'content': 'Im not interested in this bag. Im interested in the '
                              'blue one!',
                   'sender_name': 'User 1',
                   'timestamp_ms': 1675595109305},
                  {'content': 'Here is $129',
                   'sender_name': 'User 2',
                   'timestamp_ms': 1675595068468},
                  {'photos': [{'creation_timestamp': 1675595059,
                               'uri': 'url_of_some_picture.jpg'}],
                   'sender_name': 'User 2',
                   'timestamp_ms': 1675595060730},
                  {'content': 'Online is at least $100',
                   'sender_name': 'User 2',
                   'timestamp_ms': 1675595045152},
                  {'content': 'How much do you want?',
                   'sender_name': 'User 1',
                   'timestamp_ms': 1675594799696},
                  {'content': 'Goodmorning! $50 is too low.',
                   'sender_name': 'User 2',
                   'timestamp_ms': 1675577876645},
                  {'content': 'Hi! Im interested in your bag. Im offering $50. Let '
                              'me know if you are interested. Thanks!',
                   'sender_name': 'User 1',
                   'timestamp_ms': 1675549022673}],
     'participants': [{'name': 'User 1'}, {'name': 'User 2'}],
     'thread_path': 'inbox/User 1 and User 2 chat',
     'title': 'User 1 and User 2 chat'}

# loader = JSONLoader(
#     file_path=json_string,
#     jq_schema='.messages[].content',
#     text_content=True,
#     json_lines=True)

# data = loader.load()
# print(data)
# %%
json_spec = JsonSpec(dict_=json_string, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
)
json_agent_executor.run(
    "Output all the student id of the messages using a list.",
)
# %%
