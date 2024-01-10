import openai
import json
import tiktoken
import Control.context_model

from logger         import LoggerSingleton


_logger = LoggerSingleton.new_instance('log_gpt.log')

class chatgpt():
    def __init__(self, tokenID, folderID) -> None:
        self.TOKEN_GPT = tokenID
        self.TOKEN_FOLDER_ID = folderID
        openai.api_key = tokenID

    
    def gpt_post_view(self, context, gpt_model, MAX_TOLENS):
        completion = openai.chat.completions.create(
        # model="gpt-4-vision-preview",
        model=gpt_model,
        messages=context,
        max_tokens=MAX_TOLENS,
        )

        answer = Control.context_model.AnswerAssistent()
        data = json.loads( str( completion.json() ) )
        total_tokens = data['usage']['total_tokens']
        content = data['choices'][0]['message']['content']
        answer.set_answer(200, content, total_tokens)
        return answer


    def post_gpt(self, context, gpt_model):
        completion = openai.chat.completions.create(
            model=gpt_model,
            messages=context
            )

        answer = Control.context_model.AnswerAssistent()
        data = json.loads( str( completion.json() ) )
        total_tokens = data['usage']['total_tokens']
        content = data['choices'][0]['message']['content']
        answer.set_answer(200, content, total_tokens)
        return answer



    # def encode_image(self, image_path):
        # with open(image_path, "rb") as image_file:
            # return base64.b64encode(image_file.read()).decode('utf-8')





    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if not isinstance(value, str):
                    continue
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens


