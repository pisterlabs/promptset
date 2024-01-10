from .openai_chat_bot import OpenAIChatBot


def llm_init(config):
    name = config.get('name', 'openai')
    if name == 'openai':
        chat_bot = OpenAIChatBot()
        chat_bot.init(config)
        return chat_bot
    else:
        raise ValueError("Unsupported chat bot type")
