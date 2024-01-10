import os
from langchain import chat_models, prompts, callbacks
from langchain.schema import output_parser
from langsmith import Client

chain = (
  prompts.ChatPromptTemplate.from_messages(
    [
      # 'human', 'ai', or 'system'.
      ("system", "You are a conversational bot."),
      ("human", "Hello world!"),
    ]
  )
  | chat_models.ChatOpenAI()
  | output_parser.StrOutputParser()
)

with callbacks.collect_runs() as cb:
    for tok in chain.stream({"input": "Hi, I'm Clara"}):
        print(tok, end="", flush=True)
        run_id = cb.traced_runs[0].ids

client = Client()

# ... User copies the generated response
client.create_feedback(run_id, "did_copy", score=True)

# ... User clicks a thumbs up button
client.create_feedback(run_id, "thumbs_up", score=True)
