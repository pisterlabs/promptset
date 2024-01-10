from langchain.callbacks.base import BaseCallbackHandler
from io import StringIO


class StreamingCaptureCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.stdout_capture = None

    def on_streaming_output(self, output: str, **kwargs) -> None:
        if self.stdout_capture is not None:
            self.stdout_capture.write(output)

    def start(self):
        self.stdout_capture = StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self.stdout_capture
        self.stdout_capture.truncate(0)
        self.stdout_capture.seek(0)

    def stop(self):
        super().stop()
        streamed_data = self.stdout_capture.getvalue()
        self.stdout_capture.close()
        self.stdout_capture = None
        return streamed_data
