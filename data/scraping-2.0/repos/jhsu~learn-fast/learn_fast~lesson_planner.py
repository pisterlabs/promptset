import time
from openai import OpenAI
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.thread_message import Content

from typing import Optional

class LessonPlanner():
    """
    Plan a lesson based on a topic and description.
    """

    topic: str
    brain: OpenAI
    thread_id: str
    current_lesson: int

    def __init__(self, topic: str, client: OpenAI, assistant: str, thread_id: Optional[str]):
        self.topic = topic
        self.brain = client
        self.planner = self._get_or_create_assistant(assistant)
        self.current_lesson = 0
        if thread_id:
            self.thread_id = thread_id

    def prepare_lesson_plan(self):
        """
        Ask the assistant to write a lesson plan for the topic.
        """
        run = self.brain.beta.threads.create_and_run(
            assistant_id=self.planner.id,
            thread={
                "messages": [
                    {
                        "role": "user",
                        "content": f"Write a lesson plan about {self.topic}.",
                    }
                ]
            }
        )
        run = self._wait_for_run(run)
        self.thread_id = run.thread_id
        messages = self.get_thread_messages()

        last_message = messages[0]
        lesson = last_message.content[0]
        return self._get_message_text(lesson)
    
    def create_next_lesson(self):
        """
        Increment the lesson number and ask the assistant to write the next lesson.
        """
        self.current_lesson+=1
        # TODO: Check if lessons have been planned
        self.current_lesson

        self.brain.beta.threads.messages.create(self.thread_id,
            role="user", content= f"Write lesson {self.current_lesson} as a transcript with just body text.")
        self._run_thread()
        messages = self.get_thread_messages()
        content = messages[0].content[0]
        return self._get_message_text(content)

    def _get_or_create_assistant(self, id: str):
        """
        Get the OpenAI assistant with the given ID.

        TODO: Create the assistant if it doesn't exist.
        """
        return self.brain.beta.assistants.retrieve(assistant_id=id)

    def _wait_for_run(self, run: Run):
        while run.status == "queued" or run.status == "in_progress":
            run = self.brain.beta.threads.runs.retrieve(
                thread_id=run.thread_id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run


    def get_thread_messages(self):
        result = self.brain.beta.threads.messages.list(self.thread_id)
        return result.data
    
    def _run_thread(self):
        run = self.brain.beta.threads.runs.create(self.thread_id,
                                                     assistant_id=self.planner.id)
        run = self._wait_for_run(run)
        return run
    
    def _get_message_text(self, message: Content):
        if (message.type == "text"):
            return message.text.value
        raise ValueError(f"Return was unexpected type {message.type}")
