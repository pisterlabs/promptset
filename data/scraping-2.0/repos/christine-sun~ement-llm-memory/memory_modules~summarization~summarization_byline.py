## SUMMARIZATION APPROACH 2:
## Summarize the txt corpus line by line,
## modifying the summmary with each appended line

from memory import Memory
import openai
from utils import load

class SummarizationByLineMemory(Memory):
  def __init__(self, source_text, k=1):
    super().__init__(source_text)
    print("This was source text")
    print(type(source_text))
    summary = ""
    curr_lines = ""
    i = 0
    for line in source_text.splitlines():
      print("This is line")
      print(line)
      if i < len(source_text) and i < k:
        curr_lines += line
        curr_lines += "\n"
        i += 1
      # Add the curr_lines to the summary
      if i == k or i == len(source_text) - 1:
        prompt = f"""
          You are summarizing a conversation.
          This is the summary so far: \n {summary}

          And this is the new lines added in the conversation: \n {curr_lines}
          Please return the new summary for the new conversation. Ensure that the summarization is detailed and includes all relevant information about subjects in the conversation. The summary:
          """
        print("This is the prompt")
        print(prompt)
        response = openai.Completion.create(
          engine="text-davinci-003",
          prompt=prompt,
          temperature=0,
          max_tokens=250,
          stop=None,
          timeout=10
        )
        summary = response.choices[0].text.strip()
        curr_lines = ""
        i = 0

    self.summary = summary

  def query(self, query):
    prompt = f"""You are a smart, knowledgable, accurate AI with the following information:
      {self.summary}
        \nYou are sure about your answers. Please answer the following question: {query}
    """
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      temperature=0,
      max_tokens=250,
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
  source_text = load("test.txt")
  memory_test = SummarizationByLineMemory(source_text, 2)

  query ="What is Mimi's favorite physical activity?"
  answer = memory_test.query(query)
  print(query)
  print(answer)
