import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

llm = ChatOpenAI(temperature=0.0)
# memory = ConversationSummaryBufferMemory(llm=llm)

trip_plans = '''Oh, this weekend I'm actually planning a trip to Yellowstone National
Park! I'm super excited about it! I'll be starting off by driving through
the grand entrance and soaking in the stunning scenery. My first stop will
be the famous Old Faithful Geyser to catch its incredible eruptions.
Then, I'll wander around the Upper Geyser Basin on the boardwalks to 
check out all those cool geothermal features. And guess what? I'll 
be wrapping up the day with a relaxing dip in the Boiling River, 
a natural hot spring. Can't wait to unwind and enjoy the beauty of 
nature! So yeah, that's my weekend plan. How about you?'''

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

memory.save_context({"input": "Hello"}, {"output": "Hi!"})
memory.save_context({"input": "What plans do you have this weekend?"}, 
                    {"output": f"{trip_plans}"})

print(memory.load_memory_variables({}))