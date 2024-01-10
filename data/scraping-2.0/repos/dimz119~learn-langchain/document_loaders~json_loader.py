from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

# required `pip install jq`
file_path='./facebook_chat.json'
data = json.loads(Path(file_path).read_text())
pprint(data)
"""
{'image': {'creation_timestamp': 1675549016, 'uri': 'image_of_the_chat.jpg'},
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
"""

loader = JSONLoader(
    file_path='./facebook_chat.json',
    jq_schema='.messages[].content',
    text_content=False)

data = loader.load()
pprint(data)
"""
[Document(page_content='Bye!', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 1}),
 Document(page_content='Oh no worries! Bye', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 2}),
 Document(page_content='No Im sorry it was my mistake, the blue one is not for sale', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 3}),
 Document(page_content='I thought you were selling the blue one!', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 4}),
 Document(page_content='Im not interested in this bag. Im interested in the blue one!', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 5}),
 Document(page_content='Here is $129', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 6}),
 Document(page_content='', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 7}),
 Document(page_content='Online is at least $100', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 8}),
 Document(page_content='How much do you want?', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 9}),
 Document(page_content='Goodmorning! $50 is too low.', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 10}),
 Document(page_content='Hi! Im interested in your bag. Im offering $50. Let me know if you are interested. Thanks!', metadata={'source': '/Users/seungjoonlee/git/learn-langchain/document_loader/facebook_chat.json', 'seq_num': 11})]
"""