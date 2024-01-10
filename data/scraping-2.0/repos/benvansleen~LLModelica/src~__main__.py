from dotenv import load_dotenv
load_dotenv()

import functions as f
from openai_models import OpenAIMessage, OpenAIResponse
from chain import Chain
from om import om


print(om('getVersion()'))


def prompt_user(wrap_in_context: bool = False) -> OpenAIMessage:
    print('> ', end='')
    prompt = input()
    if wrap_in_context:
        prompt = Chain.wrap_prompt_in_context(prompt)
    return OpenAIMessage(
        role='user',
        content=prompt,
    )


def run_testbench(testbench: str) -> str:
    print(f'Testbench: {testbench}')
    print('Run? (y/n)')
    print('> ', end='')
    y_or_n = input().strip()
    match y_or_n:
        case 'y':
            from pyparsing.exceptions import ParseException
            try:
                return om(testbench)
            except ParseException as e:
                print(e)
                return f'{e}\nSomething went wrong... Think about it step by step...'
        case 'n':
            pass
        case _:
            print('Invalid input. Try again.')
            return run_testbench(testbench)
    return ''


def handle_response(
        response: OpenAIResponse,
        chain: Chain,
) -> Chain:
    first_choice = response.choices[0]
    chain.add(first_choice.message)

    match first_choice.finish_reason:
        case 'function_call':
            result = f.dispatch_function(response)
            if result.content.startswith('('):
                result.content = run_testbench(chain.testbench)
                print(result.content)
                result.content += '\n' + prompt_user().content
        case _:
            result = prompt_user()

    chain.add(result)
    return chain


def prompt_step(chain: Chain) -> Chain:
    if len(chain) <= 1:
        chain.add(prompt_user(wrap_in_context=True))

    if len(chain) > 10:
        chain.reload_context()

    response = f.llm(
        chain,
        model='gpt-4-0613',
        temperature=0.2,
    )
    chain.print(clear=True)
    messages = handle_response(response, chain)
    messages.print(clear=True)
    return messages


chain = Chain()
while True:
    try:
        chain = prompt_step(chain)
    except KeyboardInterrupt:
        from sys import exit
        exit()
