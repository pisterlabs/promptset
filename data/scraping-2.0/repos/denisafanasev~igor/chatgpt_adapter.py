import openai

CHAT_GPT_MODEL_NAME = "gpt-3.5-turbo-1106"


class ChatGPTAdapter:
    """
    class for chatgpt model
    """

    def __init__(self, token):
        """
        Init the chatgpt model

        Args:
            token (str): token for openai
        """        

        openai.api_key = token
    
    def get_model_name(self):
        return CHAT_GPT_MODEL_NAME

    def ask(self, message, dialog_messages=[]) -> str:
        """
        Ask the chatgpt model

        Args:
            message (_type_): _description_
            dialog_messages (list, optional): _description_. Defaults to [].

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        answer = None

        while answer is None:

            try:
                messages = self._generate_prompt_messages_for_chatgpt_api(message, dialog_messages)
                r = openai.ChatCompletion.create(
                    model=CHAT_GPT_MODEL_NAME,
                    messages=messages
                )
                answer = r.choices[0].message["content"]

            # too many tokens
            except openai.error.InvalidRequestError as error:
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from error

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]
            
            except openai.error.Timeout as error:
                #Handle timeout error, e.g. retry or log
                raise ValueError(f"OpenAI API request timed out: {error}") from error

            except openai.error.APIError as error:
                #Handle API error, e.g. retry or log
                raise ValueError(f"OpenAI API returned an API Error: {error}") from error

            except openai.error.APIConnectionError as error:
                #Handle connection error, e.g. check network or log
                raise ValueError(f"OpenAI API request failed to connect: {error}") from error
            
            except openai.error.AuthenticationError as error:
                #Handle authentication error, e.g. check credentials or log
                raise ValueError(f"OpenAI API request was not authorized: {error}") from error
            
            except openai.error.PermissionError as error:
                #Handle permission error, e.g. check scope or log
                raise ValueError(f"OpenAI API request was not permitted: {error}") from error
            
            except openai.error.RateLimitError as error:
                #Handle rate limit error, e.g. wait or log
                raise ValueError(f"OpenAI API request exceeded rate limit: {error}") from error


        answer = self._postprocess_answer(answer)

        return answer

    def _generate_prompt_messages_for_chatgpt_api(self, message, dialog_messages):

        prompt = "As an advanced chatbot named Igor, your primary goal is to assist users to the best of your ability. This may involve answering questions, providing helpful information, or completing tasks based on user input. In order to effectively assist users, it is important to be detailed and thorough in your responses but response should be not long as 4096 charts. Use examples and evidence to support your points and justify your recommendations or solutions. Remember to always prioritize the needs and satisfaction of the user. Your ultimate goal is to provide a helpful and enjoyable experience for the user. If user asks you about programming or asks to write code do it for him. If user asks about to work with the text by url, do it for him"
        messages = [{"role": "system", "content": prompt}]

        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})

        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer
