from llms import openai_wrapper


def compare_outputs(output1, output2):
    """
    Compare program outputs during testing using llm.
    To ignore minors diffs in whitespace, punctuation, etc. which is complex to do with a manual string/json comparison.

    Returns:
        bool: True if the outputs can be counted tha same, False otherwise.
    """
    msgs = [
        {"role": "system", "content": "Compare program outputs during testing. If `actual_output` is same as `expected_output` answer YES, otherwise answer NO. Ignore minor differences in whitespace, punctuation, etc. Always answer YES or NO."},
        {"role": "user", "content": f"`actual_output`: {output1} \n\n `expected_output`: {output2}"},
    ]
    res = openai_wrapper.get_chat_completion(msgs=msgs)
    if res == 'YES':
        return True
    return False


def parse_html_with_llm(html_code):
    """
    Parse important data from html code using llm.

    Returns:
        string: json parsed by llm as an example to achieve when writing the code.
    """
    msg = f"""
Parse provided html code to a dict with parsed data.
```html
{html_code}
```

We don't need technical items, ads, hidden elements, menu items, footers, etc.
We don't need headers/footers/menu, only page specific content.
You need to guess what elements can be useful for me as a user reading this page and what elements are not useful.

Keep it simple in terms of structure. Make it minimalistic, as short as possible. Add only most important fields, no need to add all possible fields, skip not critical.
Answer only with JSON, no other prose.
"""

    msgs = [
        {'role': 'user', 'content': msg}
    ]

    llm_json = openai_wrapper.get_chat_completion(msgs=msgs, response_format="json")
    return llm_json
