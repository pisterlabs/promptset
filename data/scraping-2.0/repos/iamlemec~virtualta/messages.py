# handle messages

import time
import openai

def get_content(cont):
    return '\n'.join([c.text.value for c in cont])

class MessageQueue:
    def __init__(self, assist_id, thread_id, delay=3, tries=20):
        self.client = openai.OpenAI()
        self.assist_id = assist_id
        self.thread_id = thread_id
        self.delay = delay
        self.tries = tries
        self.queue = []

    def update(self):
        # acertain last message
        if len(self.queue) == 0:
            kwargs = {}
        else:
            last, _, _ = self.queue[-1]
            kwargs = {'after': last}

        # fetch new messages
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id, order='asc', **kwargs
        )

        # append to message queue
        self.queue.extend([
            (msg.id, msg.role, get_content(msg.content)) for msg in messages
        ])

    def query(self, prompt, block=True):
        # create message
        response = self.client.beta.threads.messages.create(
            thread_id=self.thread_id, role='user', content=prompt
        )

        # run assistant
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id, assistant_id=self.assist_id,
        )

        # return if no block
        if not block:
            return

        # wait for completion
        for _ in range(self.tries):
            # get run status
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id, run_id=run.id
            )

            # check for completion
            if run.status == 'completed':
                break
            else:
                time.sleep(self.delay)
