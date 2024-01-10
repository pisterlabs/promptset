from langchain.chat_models import ChatOpenAI
import os
from senders.langchain import ObserveTracer

if not os.getenv("OPENAI_API_KEY"):
    raise Exception("You must export OPENAI_API_KEY")

tracer = ObserveTracer(log_sends=True)

llm = ChatOpenAI(temperature=0.5, model_name="gpt-4", callbacks=[tracer])
text = "What does Observe Inc do?"
print(llm.predict(text))

tracer.close()
