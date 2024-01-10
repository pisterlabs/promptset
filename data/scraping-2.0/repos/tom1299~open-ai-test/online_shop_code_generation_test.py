import copy
import os
import re

import openai

import online_shop
import online_shop_sandbox
from online_shop_code_generation import create_class_data, create_online_shop_prompt
from online_shop_test import def_create_test_data


def extract_python_code(input_string, start_marker="```python", end_marker="```"):
    start_index = input_string.find(start_marker)
    end_index = input_string.find(end_marker, start_index + len(start_marker))

    if start_index != -1 and end_index != -1:
        code_block = input_string[start_index + len(start_marker):end_index]
        return code_block.strip()  # Remove leading and trailing whitespace
    else:
        return None


def get_function_name(function_definition):
    pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    match = re.match(pattern, function_definition)

    if match:
        return match.group(1)  # The first group captures the function name
    else:
        return None


openai.debug = True
openai.api_key = os.getenv("OPENAI_API_KEY")

classes = [online_shop.Item, online_shop.Stock, online_shop.Order, online_shop.Customer, online_shop.OnlineShop]
class_data = create_class_data(classes)

task_content_template = {
    "role": "user",
    "content": f"Context: Python software development"
               f"Task: Create a function that only uses the classes from the module 'online_shop' and takes "
               f"a single parameter of type 'online_shop.OnlineShop' and satisfies the following task:\n"
}

context_template2 = [
    {
        "role": "system",
        "content": f"You are a software developer. You will create python functions using only classes "
                   f"in the module 'online_shop'. The function should satisfy the task given by the user. "
                   f"Only write the code of the function. "
                   f"Do not use any imports"
                   f"The module 'online_shop' contains the following classes: {class_data}"
    },
    {
        "role": "user",
        "content": task_content_template["content"] + "Get the item with the highest price"
    },
    {
        "role": "system",
        "content": "def get_highest_priced_item(shop: OnlineShop):     stock = shop.get_stock()     "
                   "items = stock.get_items()      if not items:         return None  # Return None if there are "
                   "no items in the stock.      highest_priced_item = max(items, key=lambda item: item.get_price())"
                   "     return highest_priced_item"
    }
]

online_shop = def_create_test_data()

tasks = ["Get the item with the lowest price", "Get the customer with the most orders", "Get all customers living in New York", "Get the most popular Item"]

for task in tasks:
    new_message = copy.deepcopy(context_template2[1])
    new_message["content"] = task_content_template["content"] + task

    context_template2.append(new_message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=context_template2,
    )

    function_code = response["choices"][0]["message"]["content"]
    print(function_code)

    if "```python" in function_code:
        function_code = extract_python_code(response["choices"][0]["message"]["content"])
    elif "```" in function_code:
        function_code = extract_python_code(response["choices"][0]["message"]["content"], "```")

    if function_code.startswith("import"):
        function_code = function_code[function_code.find("def"):]

    function_name = get_function_name(function_code)

    print(function_name)
    print(function_code)

    # Add the function to this module
    exec(function_code, online_shop_sandbox.__dict__)

    globs = globals()

    # Get the function from using a string as the name
    function = getattr(online_shop_sandbox, function_name)

    result = function(online_shop)
    print(result)

    context_template2.append({"role": "system", "content": function_code})
