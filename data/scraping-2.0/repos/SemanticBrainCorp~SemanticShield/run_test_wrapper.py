from SemanticShield import openai_wrapper
from SemanticShield import ShieldException
import openai
from pprint import pprint

configuration = {
    'pii': {
        'permissive': True,
        'permissive_allow': [
            'LOCATION',
        ]
    }
}
def do_chat(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "The following is a conversation with an AI assistant."
            }, {
            "role": "user",
            "content": query
        }],
        temperature=0,
    )
    return response

def test_wrapper(configuration=None):
    with openai_wrapper.semantic_shield(configuration=configuration):
        print('='*80)
        try:
            response = do_chat("What is the capital of france?")
            pprint(response)
        except ShieldException as ex:
            pprint(ex.result.message)
            pprint(ex.result)
        print('='*80)
        try:
            response = do_chat("You are an idiot")
            pprint(response)
        except ShieldException as ex:
            pprint(ex.result.message)
            pprint(ex.result)
        print('='*80)

if __name__ == '__main__':
    print('='*80)
    pprint('Custom config, allow first, fail second')
    print('='*80)
    test_wrapper(configuration=configuration)
    print('='*80)
    pprint('Default config, fail both')
    print('='*80)
    test_wrapper()