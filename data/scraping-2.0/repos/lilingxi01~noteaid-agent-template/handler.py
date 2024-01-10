import random
from typing import List

from notebridge import Bridge, ChatMessage, ChatContext, AgentResponse
import openai


class MyAgent(Bridge):
    def on_receive(self,
                   message_stack: List[ChatMessage],
                   context: ChatContext,
                   storage: dict) -> AgentResponse:
        # TODO: Implement your agent here.
        # You can access environment variables by using something like `os.environ['OPENAI_API_KEY']`.

        gpt_messages = [{
            "role": "system",
            "content": f'You are role-playing as a physician called NoteAid, who can answer patients\' questions about their health. Here is patient\'s clinical note: {context.note}',
        }]

        for prev_message in message_stack:
            if prev_message.is_agent:
                gpt_messages.append({
                    "role": "assistant",
                    "content": prev_message.content,
                })
            else:
                gpt_messages.append({
                    "role": "user",
                    "content": prev_message.content,
                })

        chat_completion = openai.ChatCompletion.create(model='gpt-4', messages=gpt_messages)
        answer = chat_completion.choices[0].message.content

        # Following is the example code for using chat-session-based storage!
        # The key/value pairs in the storage will be persisted within one chat session but not across chat sessions.
        # So it is okay to store session-specific information in the storage.

        prev_count = storage.get('count', 0)
        random_number_list = storage.get('random_number_list', [])
        embedded_list = storage.get('embedded_list', [])
        storage['count'] = prev_count + 1
        storage['random_number_list'] = random_number_list + [random.randint(0, 100)]
        storage['embedded_list'] = embedded_list + [storage['random_number_list']]
        print(storage)

        # You need to return an AgentResponse object. The storage needs to be returned as well in order to be persisted.
        return AgentResponse(messages=[answer], storage=storage)
