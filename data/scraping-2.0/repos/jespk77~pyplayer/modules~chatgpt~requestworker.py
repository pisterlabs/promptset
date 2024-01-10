from ui.qt import pyworker

import openai

class ChatGPTRequestWorker(pyworker.PyWorker):
    def __init__(self, owner, request):
        pyworker.PyWorker.__init__(self, owner, "ChatGPTRequestWorker")
        self._owner = owner
        self._request = request
        self._messages = []

    def run(self):
        print("VERBOSE", "Request worker started")
        request = openai.ChatCompletion.create(**self._request)
        for chunk in request:
            try:
                delta = chunk["choices"][0]["delta"]
                self._messages.append(delta if delta else chunk["choices"][0])
                if not delta: break

                content = delta.get("content", "")
                self._owner.schedule_task(task_id=self._owner.request_update_task_id, delta=content)
            except Exception as e: self.error(e, False)

    def complete(self):
        print("INFO", "Request worker completed")
        if len(self._messages) > 0:
            data = {
                "role": self._messages.pop(0).get("role", ""),
                "content": "".join([message.get("content", "") for message in self._messages])
            }

            try: reason = self._messages[-1]["finish_reason"]
            except KeyError: reason = "unknown"
            self._owner.schedule_task(task_id=self._owner.request_complete_task_id, response=data, reason=reason)

    def error(self, error, stop_and_notify=True):
        print("ERROR", "An error occurred handling ChatGPT request:", error)
        if stop_and_notify: self._owner.schedule_task(task_id=self._owner.request_error_task_id)