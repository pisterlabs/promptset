import typing
import weave
from weave import weaveflow
import example_adaptor


@weave.type()
class SingleStepModel(weaveflow.Model):
    chat_model: weaveflow.ChatModel
    example_adaptor: example_adaptor.ExampleAdaptor

    @weave.op()
    def predict(self, example: typing.Any) -> typing.Any:
        import os
        import json
        from weave.monitoring import openai

        prompt = self.example_adaptor.example_to_prompt(example)
        if self.example_adaptor.output_guidance is not None:
            prompt += "\n\n" + self.example_adaptor.output_guidance
        messages = [{"role": "user", "content": prompt}]

        response = self.chat_model.complete(messages)

        result_message = response["choices"][0]["message"]

        return {"final_answer": result_message["content"]}


@weave.type()
class SelfReflectModel(weaveflow.Model):
    chat_model: weaveflow.ChatModel
    example_adaptor: example_adaptor.ExampleAdaptor

    @weave.op()
    def predict(self, example: typing.Any) -> typing.Any:
        import os
        import json
        from weave.monitoring import openai

        prompt = self.example_adaptor.example_to_prompt(example)
        if self.example_adaptor.output_guidance is not None:
            prompt += "\n\n" + self.example_adaptor.output_guidance
        messages = [{"role": "user", "content": prompt}]

        response = self.chat_model.complete(messages)

        result_message = response["choices"][0]["message"]
        first_answer = result_message["content"]
        messages.append(result_message)

        # Reflection
        messages.append(
            {
                "role": "user",
                "content": "Now, reflecting on your prior response, do you believe it to be correct? Why or why not?",
            }
        )
        response = self.chat_model.complete(messages)
        result_message = response["choices"][0]["message"]
        reflection = result_message["content"]
        messages.append(result_message)

        # Final response
        second_prompt = "OK, given your prior response and your self reflection, please provide a final answer."
        if self.example_adaptor.output_guidance is not None:
            second_prompt += " " + self.example_adaptor.output_guidance
        messages.append({"role": "user", "content": second_prompt})
        response = self.chat_model.complete(messages)
        result_message = response["choices"][0]["message"]
        second_answer = result_message["content"]

        return {
            "first_answer": first_answer,
            "reflection": reflection,
            "final_answer": second_answer,
        }
