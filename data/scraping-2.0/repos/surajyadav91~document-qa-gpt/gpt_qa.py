import openai

openai.api_key = open("key.txt", "r").read()


class GPT_QA:
    def __init__(self,config):
        self.chat_gpt_model = config['chat_gpt_model']

    def read(self, question, passage):
        messages = [
            dict(
                role="system",
                content="you are an intelligent question-answering assistant, you will answer the questions based on context text fed to you.",
            )
        ]

        messages.append(dict(role="user", content=passage))
        messages.append(
            dict(
                role="user",
                content=f"\n\n Based on the above text answer this question \n\n, question: {question}",
            )
        )

        response = openai.ChatCompletion.create(
            model=self.chat_gpt_model, messages=messages
        )

        return response["choices"][0]["message"]["content"]


