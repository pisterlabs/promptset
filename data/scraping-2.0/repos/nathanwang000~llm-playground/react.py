# This code is Apache 2 licensed:
# https://www.apache.org/licenses/LICENSE-2.0
# adapted from https://til.simonwillison.net/llms/python-react-pattern
import openai
import re
import httpx
import os
import click
from lib.utils import ChatBot, repl

prompt = """
You run in a loop of Thought, Action, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you
Observation will be the result of running those actions.

Your available actions are:

{tools}

Make sure your output "Action: <action>: ```<query>```" on a single line.

Example session:

Question: What is the capital of France?
Thought: I should look up France on Wikipedia
Action: wikipedia: ```France```

You will be called again with this:

Observation: France is a country. The capital is Paris.

You then output:

Answer: The capital of France is Paris
""".strip()

action_re = re.compile('^Action: (\w+): ```([^`]*)```.*$')

def query(question, max_turns=5):
    i = 0
    bot = ChatBot(prompt,
                  stop=["Observation:", "observation"] )
    next_prompt = question
    while i < max_turns:
        print(f"turn: {i}")
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                error_msg = "Unknown action: {}: {}".format(action, action_input)
                # raise Exception(error_msg)
                print('Error', error_msg)
                return
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}\nQuestion: {}".format(observation, question)
        else:
            break

def wikipedia(q):
    '''
    Returns a summary from searching Wikipedia
    '''
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

def ddgs_text(q, top_n=3):
    '''
    Returns top results from duckduckgo, useful if you want to look something up online
    '''
    from duckduckgo_search import DDGS
    ddgs = DDGS()
    ddgs_gen = ddgs.text(q, safesearch='off')
    return "\n".join([str(r) for i, r in enumerate(ddgs_gen) if i < top_n])

def calculate(what):
    '''
    Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary. The input must be executable python code.
    '''
    from sympy import simplify
    return simplify(what)

known_actions = { # like load tools for langchain
    "wikipedia": wikipedia,
    "calculate": calculate,
    "search": ddgs_text
}

# load available tools
tools = []
for k, v in known_actions.items():
    tools.append("\n".join(["{}: \ne.g., {}: ```<query>``` \n{}".format(k, k,
                                                                  v.__doc__.strip())]))
prompt = prompt.format(tools="\n\n".join(tools))

@click.command()
@click.option('-q', 'question', prompt='Question', help='Question to ask')
def main(question):
    click.echo(query(question))
    
if __name__ == "__main__":
    main()
