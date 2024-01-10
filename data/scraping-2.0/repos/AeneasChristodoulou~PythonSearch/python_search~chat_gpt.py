import os


from python_search.error.exception import notify_exception

SUPPORTED_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "text-davinci-003",
    "curie:ft-jean-personal-2023-03-20-21-40-47",
]


class LLMPrompt:
    """
    Uses OpenAI to answer a given prompt.
    """

    MODEL_ENGINE = "text-davinci-003"

    def __init__(self, max_tokens=500):
        self.max_tokens = int(max_tokens)

    def collect_prompt_via_ui(self):
        """
        Collects a prompt from the user via a UI.

        :return:
        """
        from python_search.apps.collect_input import CollectInput

        message = CollectInput().launch()
        self.answer(message)

    def given_prompt_plus_clipboard(self, given_prompt, return_promt=True):
        """
        Appends clipboard content to the given prompt and returns a string with the result.

        :param given_prompt:
        :return:
            str: A string with the result of combining the prompt with clipboard content.
        """

        from python_search.apps.clipboard import Clipboard

        content = Clipboard().get_content()

        prompt = f"{given_prompt}: {content}"
        result = self.answer(prompt)

        prompt_str = f"""
-------
Prompt:
{prompt}
"""

        if return_promt:
            result = f"""
{result}

{prompt_str}
            """
        else:
            result

        print(result)

    @notify_exception()
    def answer(self, prompt: str, debug=False, max_tokens=500, model=None) -> str:
        """
        Answer a prompt with openAI results
        """
        if len(prompt) > 4097:
            prompt = prompt[:4097]
        self.max_tokens = int(max_tokens)

        if model == "gpt-3.5-turbo":
            return ChatAPI().prompt(prompt)

        import openai

        openai.api_key = os.environ["OPENAI_KEY"]
        # Set the maximum number of tokens to generate in the response

        if debug:
            print("Prompt: ", prompt)

        engine = self.MODEL_ENGINE
        if model is not None:
            engine = None

        try:
            print("Open AI Key used: ", openai.api_key)
            # Generate a response
            completion = openai.Completion.create(
                model=model,
                engine=engine,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return completion.choices[0].text.strip()
        except openai.error.RateLimitError as e:
            print(str(e))
            from python_search.apps.notification_ui import send_notification

            send_notification(str(e))
            return ""
        return ""

        # Print the response

    def print_answer(self, prompt):
        print(self.answer(prompt))


class ChatAPI:
    def prompt(self, text) -> str:
        import openai

        openai.api_key = os.environ["OPENAI_KEY"]
        #print("Open AI Key used: ", openai.api_key)

        try:
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": text},
                ],
            )
            return result.choices[0].message.content

        except openai.error.RateLimitError as e:
            print(str(e))
            from python_search.apps.notification_ui import send_notification

            send_notification(str(e))
            return ""

        return ""


def main():
    import fire

    fire.Fire()


if __name__ == "__main__":
    main()
