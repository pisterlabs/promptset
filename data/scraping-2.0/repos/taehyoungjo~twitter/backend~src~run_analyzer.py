"""
Ask natural language questions about the run data!
We load a file which contains the run data.

Each line is a JSON array, with objects with fields:

- type: "TWEET", "QUOTE", "COMMENT"
- tweet_id: int
- user_id: int
- likes: list[int] (user ids)
- retweets: list[int] (user ids)
- quotes: list[int] (tweet ids)
- comments: list[int] (tweet ids)
- timestamp: int
- content: str
- parent_id: Optional[int] (tweet id)
"""

# %%
import argparse
import json
import os

from config import Tweet, TweetType
from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage
from pydantic import BaseModel

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class TweetList(BaseModel):
    __root__: list[Tweet]


parser = argparse.ArgumentParser(description="Process some integers.")

file_path = "../runs/tweets-a3422d1c-cc31-4434-a653-41f0c6bf417e.json"

states: list[TweetList] = []
with open(file_path) as f:
    for line in f:
        state = TweetList.parse_obj(json.loads(line))
        states.append(state)

# convert states to list[list[Tweet]]
states = [state.__root__ for state in states]

llm = ChatAnthropic(
    anthropic_api_key=ANTHROPIC_API_KEY,
    model="claude-2",
    temperature=0.2,
    max_tokens_to_sample=4096,
)

query = "Plot the total comments the user with user_id = 2 received on their posts over time."

result = llm.generate(
    [
        [
            HumanMessage(
                content="""\
You are an expert Python programmer and data scientist. You know how to use matplotlib to plot data. You have a list of list of tweets. Here are the datatypes:
```python
class TweetType(str, Enum):
    TWEET = "TWEET"
    QUOTE = "QUOTE"
    COMMENT = "COMMENT"  # not shown in timeline?

class Tweet(BaseModel):
    type: TweetType
    tweet_id: int = -1  # sentinel
    user_id: int  # person who made the tweet
    likes: list[int] = Field(default_factory=list)  # list of user_ids who liked
    retweets: list[int] = Field(default_factory=list)  # list of user_ids who retweeted
    quotes: list[int] = Field(default_factory=list)  # list of tweet_ids that are quotes
    comments: list[int] = Field(
        default_factory=list
    )  # list of tweet_ids that are replies
    timestamp: int  # unix timestamp, not using for now
    content: str  # the actual text of the tweet
    parent_id: Optional[int] = None  # only if type is QUOTE or COMMENT

states: list[list[Tweet]] # list of list of tweets, each list is a timestep
```

Give me Python code that will plot the total comments the user with user_id = 2 received on their posts over time. Write the code in a markdown code block labelled as Python. Put a detailed comment before every line explaining your work.\
"""
            ),
            AIMessage(
                content="```python\n",
            ),
        ]
    ],
    stop=["```"],
)

print(result.generations[0][0].text)

exec(result.generations[0][0].text.strip("```"))

# %%
