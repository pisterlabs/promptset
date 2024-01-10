from openai_afc import AutoFnChatCompletion, AutoFnDefinition, AutoFnParam
from openai_afc.errors import GPTSchemaDeviation

def add(x, y):
    print(f'Adding {x} and {y}')
    return x + y

def multiply(x, y):
    print(f'Multiplying {x} and {y}')
    return x * y

funcs = [
    AutoFnDefinition(add, description='add two numbers. each parameter must be a single number, not an expression.', params=[
        AutoFnParam('x', {'type': 'number'}),
        AutoFnParam('y', {'type': 'number'})
    ]),
    AutoFnDefinition(multiply, description='multiply two numbers. each parameter must be a single number, not an expression.', params=[
        AutoFnParam('x', {'type': 'number'}),
        AutoFnParam('y', {'type': 'number'})
    ])
]

# GPT 3.5 likes to try and pass 175912 * 17 as a JSON number itself
try:
    completion = AutoFnChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What is 851512 + 175912 * 17?"}
        ],
        functions=funcs,
        temperature=0.5
    )
    print(completion['choices'][0]['message']['content'])
except GPTSchemaDeviation as e:
    print(e)

# Adding parentheses and system message seems to help GPT understand that these need to be distinct operations
completion = AutoFnChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Each math operation must be done individually"},
        {"role": "user", "content": "What is 851512 + (175912 * 17)?"}
    ],
    functions=funcs,
    temperature=0.5
)
print(completion['choices'][0]['message']['content'])