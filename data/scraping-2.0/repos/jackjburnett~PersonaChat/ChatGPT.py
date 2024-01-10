import llm
from openai.error import RateLimitError


def GPT35Call(prompt):
    model = llm.get_model("gpt-3.5-turbo")
    model.key = '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'
    try:
        gpt_response = model.prompt(prompt)
        response = gpt_response.text()
    except RateLimitError:
        response = "OpenAI API Rate Limit Error"
    return response
