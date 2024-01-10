
import openai

from util import current_time_ms
from wandb.sdk.data_types.trace_tree import Trace

class GTP4LLM:
    def __init__(self, api_token) -> None:
        self.reset_history()
        self.api_token = api_token

    def reset_history(self) -> None:
        self.chat_history = [ 
            {"role": "system", "content": "You are a children's book author helping me write a children's book for my friend's children"}
        ]

    def generate_text(self, prompt, wandb_parent_span, model_name="gpt-4-1106-preview"):
        # Append the user's prompt to the chat history
        self.chat_history.append({"role": "user", "content": prompt})

        # Call the GPT-4 model
        openai.api_key = self.api_token
        # model_name = "gpt-4"
        # model_name = "gpt-4-1106-preview"
        start_time = current_time_ms()

        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=self.chat_history,
                # functions=functions,
            )

            end_time = current_time_ms()
            status = "success"
            status_message = "Success"
            response_text = response["choices"][0]["message"]["content"]
            token_usage = response["usage"].to_dict()

        except Exception as e:
            end_time = current_time_ms()
            status = "error"
            status_message = str(e)
            response_text = ""
            token_usage = {}


        llm_prompt_span = Trace(
            name="LLMPrompt",
            kind="llm",
            status_code=status,
            status_message=status_message,
            metadata={
                "token_usage": token_usage,
                "model_name": model_name,
            },
            start_time_ms=start_time,
            end_time_ms=end_time,
            inputs={ "prompt": prompt, "chat_history": self.chat_history[:-1]},
            outputs={"response": response_text},
        )
        wandb_parent_span.add_child(llm_prompt_span)
        wandb_parent_span.add_inputs_and_outputs(
            inputs={"prompt": prompt}, outputs={"response": response_text}
        )
        wandb_parent_span._span.end_time_ms = end_time

        if status == "success":
            # Extract and append the model's reply to the chat history
            self.chat_history.append({"role": "assistant", "content": response_text})

        return response_text.strip()
