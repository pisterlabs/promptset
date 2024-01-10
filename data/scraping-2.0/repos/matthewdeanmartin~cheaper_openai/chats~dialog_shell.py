# from openai.types.chat import ChatCompletionMessageParam
# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))

# async def completion(self, messages: list[ChatCompletionMessageParam]):
#     messages = [
#         {
#             "role": "user",
#             "content": "How do I do the thing?",
#         },
#     ],
#     completion = await self.client.chat.completions.create(model=self.model, messages=messages)
#     choice = completion.choices[0]
#     print(choice.message.content)
#     return completion
