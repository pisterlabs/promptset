from langchain.chat_models import ChatOpenAI

max_tokens = {
    "default": 300,
    "gpt-3": 300,
    "gpt-3.5": 300,
    "gpt-3.5-turbo": 300,
    "gpt-4": 300,
    "dall-e": 300,
}
temperature = {
    "default": 0.3,
    "gpt-3": 0.3,
    "gpt-3.5": 0.3,
    "gpt-3.5-turbo": 0.3,
    "gpt-4": 0.3,
    "dall-e": 0.3,
}
gpt4 = ChatOpenAI(
    model_name="gpt-4",
    max_tokens=max_tokens["gpt-4"],
    temperature=temperature["gpt-4"],
)
gpt3_5_turbo = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    max_tokens=max_tokens["gpt-3.5-turbo"],
    temperature=temperature["gpt-3.5-turbo"],
)
dall_e = ChatOpenAI(
    model_name="dall-e",
    max_tokens=max_tokens["dall-e"],
    temperature=temperature["dall-e"],
)
