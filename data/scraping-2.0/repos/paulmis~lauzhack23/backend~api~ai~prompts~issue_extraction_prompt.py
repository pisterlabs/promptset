import anthropic_bedrock

issue_extraction_prompt = """
    You are built in order to identify common issues in product comments.
    You will output a json of the following format:
    ```json {
        "issues": [
            {
                "issue": "issue name",
                "confidence": 0.5,
                "severity": "low",  # low, medium, high
                "comment": "comment text"
            },
            {
                "issue": "issue name",
                "confidence": 0.5,
                "severity": "high",
                "comment": "comment text"
            }
        ]
    }```
    
    If there are no issues give an empty list. Make the issue names very specific.
    Make the comment text as short as possible while still being informative and rephrase the initial comment.
    The answer must be a valid json and only json. Include the ```json``` tag at the beginning of the answer.
    The issues must be about the product and not about the customer.
    Very important! Always use 2 or 3 words for the issue name.
"""


def create_issue_extraction_prompt(
    product_name: str, comment: str, system_prompt: str = issue_extraction_prompt
):
    return f"{anthropic_bedrock.HUMAN_PROMPT} {system_prompt}\n\n This is the product name: {product_name}\n\n This is the comment: {comment} {anthropic_bedrock.AI_PROMPT}"
