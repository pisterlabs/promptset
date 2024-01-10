import pprint

from code_generator.model.function_call import FunctionCall


class EvaluatedCompletion:
  """
  A class that encapsulates the parsed and processed completion, along with
  information about errors or issues it produces

    Attributes:
        content (str): Completion from a model.
        issues (List[str]): Information about issues related to this
        content (if the completion has produced some invalid result on a
        previous step).
    """

  def __init__(self, completion_response):
    """
    Initializes a new EvaluatedCompletion object.

    Args:
        completion_response (dict): the response from OpenAI content API.
    """
    function_call_obj = extract_function_call(completion_response)
    self.content = extract_content(completion_response)
    if function_call_obj:
      self.function_call = FunctionCall(function_call_obj)
    else:
      self.function_call = None
    self.issues = []

  def __str__(self):
    """
    Returns a string representation of the EvaluatedCompletion object.

    Returns:
      str: A string representation of the object.
    """
    issues_str = "\n".join(
      [f"Issue {i + 1}: {issue}" for i, issue in enumerate(self.issues)])
    if self.content:
      return f"Content: {self.content}\n{issues_str}"
    elif self.function_call:
      return f"Function: {self.function_call.name}\n{issues_str}"

  def to_messages(self):
    """
    Converts the EvaluatedCompletion into a list of messages.

    Returns:
      list: A singleton list of assistant message.
    """
    if self.content:
      return [{"role": "assistant", "content": self.content}]
    elif self.function_call:
      return [{"role": "assistant", "content": self.content, "function_call":
              self.function_call.to_payload()}]

def extract_finish_reason(completion_response):
  return completion_response['choices'][0]['finish_reason']


def extract_content(completion_response):
  return completion_response['choices'][0]['message'].get('content', None)


def extract_function_call(completion_response):
  return completion_response['choices'][0]['message'].get('function_call', None)
