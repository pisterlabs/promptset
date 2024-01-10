import openai


class ChatAPI:
    """
    Class to manage interactions with ChatGPT API.
    Instantiate using ChatGPT API Key.
    """
    def __init__(self, api_key):
        openai.api_key = api_key

    @staticmethod
    def get_chat_response(message: str,
                          model="gpt-3.5-turbo",
                          max_tokens=999,
                          temperature=0.5 ) -> (str, str):
        """
        Call openAI API to send message and receive response.
        :param temperature: What sampling temperature to use, between 0 and 2.
        :param max_tokens:  The maximum number of tokens to generate in the completion.
        :param model:  ID of the model to use
        :param message: a string with the message to send to ChatGPT
        :return: the response received (or None)
        """
        if message:
            chat = openai.ChatCompletion.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": message}]
            )
            stats = f'model: {chat.model}\ttokens: {str(chat.usage.total_tokens)} ' \
                    f'({str(chat.usage.prompt_tokens)}/{str(chat.usage.completion_tokens)})'
            response = ''
            for choice in chat.choices:
                response += choice.message.content + '\n'
            return response, stats
        else:
            return None, None

    def get_model_names(self) -> [str]:
        models = openai.Model.list()
        model_names = []
        for model in models.data:
            model_names.append(model.id)
        return model_names