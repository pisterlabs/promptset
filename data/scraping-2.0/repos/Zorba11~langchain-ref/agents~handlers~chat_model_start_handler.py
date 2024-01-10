from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import BaseMessage
from pyboxen import boxen

def boxen_print(*args, **kwargs):
  print(boxen(*args, **kwargs))

class ChatModelStartHandler(BaseCallbackHandler):
  def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
    print("\n\n===== Sending Messages =========\n\n")

    for message in messages[0]:
      if message.type == "system":
        boxen_print(message.content, title=message.type, color="yellow")

      elif message.type == "human":
        boxen_print(message.content, title=message.type, color="green")

      elif message.type == "ai" and "function_call" in message.additional_kwargs:
        call = message.additional_kwargs["function_call"]
        boxen_print(f"Running tool {call['name']} with args {call['arguments']}", title=message.type, color="cyan")

      elif message.type == "ai":
        boxen_print(message.content, title=message.type, color="blue")

      elif message.type == "function":
        boxen_print(message.content, title=message.type, color="purple")

      else:
        boxen_print(message.content, title=message.type, color="red")