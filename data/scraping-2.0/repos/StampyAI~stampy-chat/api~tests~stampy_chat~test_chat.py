from unittest.mock import patch
from langchain.llms.fake import FakeListLLM
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import ChatMessage, HumanMessage, SystemMessage

from stampy_chat.settings import Settings
from stampy_chat.callbacks import StampyCallbackHandler
from stampy_chat.chat import (
    LimitedConversationSummaryBufferMemory,
    MessageBufferPromptTemplate,
    PrefixedPrompt,
    make_memory,
    merge_history,
)


def make_prompt_template(max_tokens, examples):
    template = ChatPromptTemplate.from_template('{content}')
    return MessageBufferPromptTemplate(
        example_prompt=template,
        get_num_tokens=lambda s: len(s),
        max_tokens=max_tokens,
        examples=examples,
    )


def test_MessageBufferPromptTemplate_format_messages():
    template = make_prompt_template(100, [
        {'content': 'bla bla bla'},
        {'content': 'ble ble ble'},
        {'content': 'and some more'},
    ])
    assert template.format_messages() == [
        HumanMessage(content='bla bla bla'),
        HumanMessage(content='ble ble ble'),
        HumanMessage(content='and some more'),
    ]


def test_MessageBufferPromptTemplate_format_messages_truncated():
    template = make_prompt_template(20, [
        {'content': 'bla bla bla'},
        {'content': 'ble ble ble'},
        {'content': 'and some more'},
    ])
    assert template.format_messages() == [
        HumanMessage(content='bla bla bla'),
    ]


def test_MessageBufferPromptTemplate_format_all_messages_truncated():
    template = make_prompt_template(10, [
        {'content': 'bla bla bla'},
        {'content': 'ble ble ble'},
        {'content': 'and some more'},
    ])
    assert template.format_messages() == []


def test_PrefixedPrompt_format_messages():
    prompt = PrefixedPrompt(messages_field='history', prompt='bla bla bla', input_variables=[])
    history = [HumanMessage(content=f'human message {i}') for i in range(5)]
    assert prompt.format_messages(history=history) == [
        SystemMessage(content='bla bla bla'),
        HumanMessage(content='human message 0'),
        HumanMessage(content='human message 1'),
        HumanMessage(content='human message 2'),
        HumanMessage(content='human message 3'),
        HumanMessage(content='human message 4'),
    ]


def test_PrefixedPrompt_format_messages_no_history():
    prompt = PrefixedPrompt(messages_field='history', prompt='bla bla bla', input_variables=[])
    assert prompt.format_messages(history=[]) == []


def test_LimitedConversationSummaryBufferMemory_set_empty():
    llm = FakeListLLM(responses=['this is a summary of what was before'])
    memory = LimitedConversationSummaryBufferMemory(llm=llm)
    memory.chat_memory = [{'content': 'bla bla bla', 'role': 'human'}]

    memory.set_messages([])
    assert memory.chat_memory == ChatMessageHistory(messages=[])


def test_LimitedConversationSummaryBufferMemory_set():
    llm = FakeListLLM(responses=['this is a summary of what was before'])
    memory = LimitedConversationSummaryBufferMemory(llm=llm)

    memory.set_messages([
        {'content': 'a system message', 'role': 'system'},
        {'content': 'bla bla bla', 'role': 'human'},
    ])
    assert memory.chat_memory == ChatMessageHistory(messages=[
        ChatMessage(content='a system message', role='system'),
        ChatMessage(content='bla bla bla', role='human'),
    ])


def test_LimitedConversationSummaryBufferMemory_set_more():
    llm = FakeListLLM(responses=['this is a summary of what was before'])
    memory = LimitedConversationSummaryBufferMemory(llm=llm, max_history=4)

    memory.set_messages([
        {'content': 'a system message', 'role': 'system'},
        {'content': 'message 1 - should be summarized', 'role': 'human'},
        {'content': 'message 2 - should be summarized', 'role': 'human'},
        {'content': 'message 3 - should be kept', 'role': 'human'},
        {'content': 'message 4 - should be kept', 'role': 'human'},
        {'content': 'message 5 - should be kept', 'role': 'human'},
    ])
    assert memory.chat_memory == ChatMessageHistory(messages=[
        ChatMessage(content='this is a summary of what was before', role='assistant'),
        ChatMessage(content='message 3 - should be kept', role='human'),
        ChatMessage(content='message 4 - should be kept', role='human'),
        ChatMessage(content='message 5 - should be kept', role='human'),
    ])


def test_LimitedConversationSummaryBufferMemory_set_with_callbacks():
    callback_calls = {}
    class DummyCallback(StampyCallbackHandler):
        def on_memory_set_start(self, history):
            callback_calls['start'] = history

        def on_memory_set_end(self, messages):
            callback_calls['end'] = messages

    llm = FakeListLLM(responses=['this is a summary of what was before'])
    memory = LimitedConversationSummaryBufferMemory(llm=llm, callbacks=[DummyCallback()])
    history = [
        {'content': 'a system message', 'role': 'system'},
        {'content': 'bla bla bla', 'role': 'human'},
    ]

    memory.set_messages(history)
    assert memory.chat_memory == ChatMessageHistory(messages=[
        ChatMessage(content='a system message', role='system'),
        ChatMessage(content='bla bla bla', role='human'),
    ])
    assert callback_calls == {
        'start': history,
        'end': memory.chat_memory,
    }


def test_make_memory_skips_deleted():
    history = [
        {'content': 'this should be kept', 'role': 'system'},
        {'content': 'as should this', 'role': 'human'},
        {'content': 'this will be ignored', 'role': 'deleted'},
        {'content': 'bla bla bla', 'role': 'assistant'},
        {'content': 'remove me!!', 'role': 'deleted'},
    ]
    with patch('stampy_chat.chat.get_model', return_value=FakeListLLM(responses=[])):
        mem = make_memory(Settings(), history, [])
    assert mem.chat_memory == ChatMessageHistory(messages=[
        ChatMessage(content='this should be kept', role='system'),
        ChatMessage(content='as should this', role='human'),
        ChatMessage(content='bla bla bla', role='assistant'),
    ])


def test_merge_history_empty():
    assert merge_history([]) == []


def test_merge_history_no_merges():
    history = [
        {'content': 'this should be kept', 'role': 'system'},
        {'content': 'as should this', 'role': 'human'},
        {'content': 'this will be ignored', 'role': 'deleted'},
        {'content': 'bla bla bla', 'role': 'assistant'},
        {'content': 'remove me!!', 'role': 'deleted'},
    ]
    assert merge_history(history) == history


def test_merge_history_merges():
    history = [
        {'role': 'user', 'content': 'question 1'},
        {'role': 'assistant', 'content': 'answer 1 part 1'},
        {'role': 'assistant', 'content': 'answer 1 part 2'},
        {'role': 'user', 'content': 'question 2 part 1'},
        {'role': 'user', 'content': 'question 2 part 2'},
        {'role': 'user', 'content': 'question 2 part 3'},
        {'role': 'assistant', 'content': 'answer 2'},
        {'role': 'user', 'content': 'question 3'},
        {'role': 'assistant', 'content': 'answer 3'},
        {'role': 'user', 'content': 'question 4 part 1'},
        {'role': 'user', 'content': 'question 4 part 2'},
    ]
    assert merge_history(history) == [
        {'role': 'user', 'content': 'question 1'},
        {'role': 'assistant', 'content': 'answer 1 part 1\nanswer 1 part 2'},
        {'role': 'user', 'content': 'question 2 part 1\nquestion 2 part 2\nquestion 2 part 3'},
        {'role': 'assistant', 'content': 'answer 2'},
        {'role': 'user', 'content': 'question 3'},
        {'role': 'assistant', 'content': 'answer 3'},
        {'role': 'user', 'content': 'question 4 part 1\nquestion 4 part 2'},
    ]
