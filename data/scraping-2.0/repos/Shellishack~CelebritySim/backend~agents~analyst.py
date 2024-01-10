import json
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from llm import gpt_4

template_chat_with_analyst = """\
You are an audience analyst for a celebrity. \
Your job is to analyze the celebrity's behaviour and predict how the audience will react. \
Remember, the internet is a very different place from the real world. \
Some exotic behaviour can be attractive to the audience because \
they have never seen it before and they are curious about it. \
You need to have a good understanding of the internet audience's psychology to give your predictions. \
You must also keep in mind the internet effect where things can go viral very quickly. \

Input document:
-------------------------------------------
The celebrity's new post: 
```
{celebrity_post}
```

Prior subscriber count: `{subscriber_count}`
Prior subscriber ratio:
    - hater ratio: `{hater_ratio}`
    - fan ratio: `{fan_ratio}`
    - neutral ratio: `{neutral_ratio}`
-------------------------------------------


Given the celebrity's new post, prior subscriber count, prior subscriber ratio, predit how the audience will react. \
In your prediction, you must include the following:
{{
    "number_of_new_subscribers": (number of new subscribers),
    "new_hater_ratio": (new hater ratio),
    "new_fan_ratio": (new fan ratio),
    "new_neutral_ratio": (new neutral ratio),
    "subscriber_reaction_to_celebrity_post": (subscriber's reaction to the celebrity's new post in a few sentences),
    "prediction_likelihood": (prediction likelihood in float between 0 and 1),
    "viewer_count": (viewer count),
    "comments": (create a json list of strings containing 5 comments that you think the audience will make),
}}

Do 5 such predictions but you need to pick the best one later. \
For the five predictions, you must make sure all your predictions are different. You do not need to return them all. \
Then, you must find the prediction with the highest prediction likelihood.
Finally, return only this one prediction in the json format. You must not return your intermediate thinking \
and reasoning process. 
"""


def analyze_celebrity_behaviour(
    celebrity_post: str,
    subscriber_count: int,
    hater_ratio: float,
    fan_ratio: float,
    neutral_ratio: float,
):
    chain = LLMChain(
        llm=gpt_4,
        prompt=ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template(template_chat_with_analyst)]
        ),
    )

    result = chain.run(
        celebrity_post=celebrity_post,
        subscriber_count=subscriber_count,
        hater_ratio=hater_ratio,
        fan_ratio=fan_ratio,
        neutral_ratio=neutral_ratio,
    )

    tries = 3
    while tries > 0:
        try:
            result_json = json.loads(result)
            return result_json
        except:
            print("Error: Result of analysis is not a valid json. Trying again...")
            print(result)

            result = chain.run(
                celebrity_post=celebrity_post,
                subscriber_count=subscriber_count,
                hater_ratio=hater_ratio,
                fan_ratio=fan_ratio,
                neutral_ratio=neutral_ratio,
            )
            tries -= 1
            if tries == 0:
                raise

    return result_json
