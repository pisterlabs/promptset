"""Quick Replies assistant functions"""
import openai


def parse_prompt(file: str) -> str:
    """Loads prompts for quick replies"""
    with open(file, "r", encoding="utf-8") as promptfile:
        prompt = promptfile.read()
    return prompt


async def generate_quick_replies(message: str) -> list[str]:
    """Generates quick replies"""
    prompt = parse_prompt("api/assistants/quick_replies/quick_replies.txt")
    # Input gpt response
    messages = [
        {"role": "system", "content": prompt},
        {"role": "assistant", "content": message},
    ]

    # Generate open ai response
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    # Extract response message as string
    response_message = gpt_response["choices"][0].message["content"]
    # Split string into list of sub str and return
    quick_replies = response_message.split("//")
    return quick_replies
