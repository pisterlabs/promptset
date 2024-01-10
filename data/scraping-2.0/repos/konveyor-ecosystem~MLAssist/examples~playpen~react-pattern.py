import re
import httpx

from langchain.llms import LlamaCpp
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamingStdOutCallbackHandler

import dotenv

dotenv.load_dotenv()

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# llm = LlamaCpp(
#   model_path="../models/wizardlm-13b-v1.1.ggmlv3.q2_K.bin",
#   callback_manager=callback_manager,
#   verbose=False,
#   max_tokens=10000
# )

llm = ChatOpenAI(temperature=0.0)

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer and then PAUSE
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

wikipedia:
e.g. wikipedia: Django
Returns a summary from searching Wikipedia. Always look things up on Wikipedia if you have the opportunity to do so.

simon_blog_search:
e.g. simon_blog_search: Django
Search Simon's blog for that term

Session:

Question: What is the capital of France?
Thought: I should look up France on Wikipedia
Action: wikipedia: France
PAUSE
Observation: France is a country. The capital is Paris.
Answer: The capital of France is Paris
PAUSE
""".strip()


action_re = re.compile('^Action: (\w+): (.*)$')

def query(question, max_turns=5):
  i = 0
  # bot = ChatBot(prompt)
  # next_prompt = question

  next_prompt = prompt + "\nQuestion: " + question + "\n"
  while i < max_turns:
    i += 1
    # result = bot(next_prompt)
    print("PROMPT GIVEN:")
    print(next_prompt)
    result = llm.predict(next_prompt, stop=['PAUSE', 'Question:'])
    print("RESULT:")
    print(result)
    actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
    if actions:
      print("ACTION FOUND")
      # There is an action to run
      action, action_input = actions[0].groups()
      if action not in known_actions:
        raise Exception("Unknown action: {}: {}".format(action, action_input))
      print(" -- running {} {}".format(action, action_input))
      observation = known_actions[action](action_input)
      print("Observation:", observation)
      next_prompt = next_prompt + "\nPAUSE\nObservation: {}\n".format(observation)
    else:
      print("NO ACTIONS")
      return


def wikipedia(q):
  return httpx.get("https://en.wikipedia.org/w/api.php", params={
    "action": "query",
    "list": "search",
    "srsearch": q,
    "format": "json"
  }).json()["query"]["search"][0]["snippet"]


def simon_blog_search(q):
  results = httpx.get("https://datasette.simonwillison.net/simonwillisonblog.json", params={
    "sql": """
    select
      blog_entry.title || ': ' || substr(html_strip_tags(blog_entry.body), 0, 1000) as text,
      blog_entry.created
    from
      blog_entry join blog_entry_fts on blog_entry.rowid = blog_entry_fts.rowid
    where
      blog_entry_fts match escape_fts(:q)
    order by
      blog_entry_fts.rank
    limit
      1""".strip(),
    "_shape": "array",
    "q": q,
  }).json()
  return results[0]["text"]

def calculate(what):
  return eval(what)

known_actions = {
  "wikipedia": wikipedia,
  "calculate": calculate,
  "simon_blog_search": simon_blog_search
}

while True:
  query(input("> "))