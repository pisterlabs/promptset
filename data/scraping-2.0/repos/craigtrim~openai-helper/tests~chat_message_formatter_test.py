#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from openai_helper.dmo import ChatMessageFormatter


def test_service():

    formatter = ChatMessageFormatter().process
    assert formatter

    result = formatter(
        input_prompt='You are a helpful assistant.',
        messages=[
            'Who won the world series in 2020?',
            'The Los Angeles Dodgers won the World Series in 2020.',
            'Where was it played?'])

    print(result)
    assert result == [
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': 'Who won the world series in 2020?'
        },
        {
            'role': 'assistant',
            'content': 'The Los Angeles Dodgers won the World Series in 2020.'
        },
        {
            'role': 'user',
            'content': 'Where was it played?'
        }
    ]


def main():
    test_service()


if __name__ == '__main__':
    main()
