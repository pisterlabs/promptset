import anthropic_bedrock

highlight_prompt = """
    You will be given a product review and an issue identified in the product that is mentioned in the review.
    Your task is to give as json the exact segment where the issue is mentioned in the review.
    The text should be exactly as it appears in the review.
    The answer must be a valid json and only json. Include the ```json``` tag at the beginning of the answer.
    
    Example:
    ```json
    {
        "text": "comment text"
    }
    ```
"""


def create_highlight_prompt(
    product_review: str, issue: str, system_prompt: str = highlight_prompt
):
    return f"{anthropic_bedrock.HUMAN_PROMPT} {system_prompt}\n\n This is the product review: {product_review}\n\n This is the issue: {issue} {anthropic_bedrock.AI_PROMPT}"
