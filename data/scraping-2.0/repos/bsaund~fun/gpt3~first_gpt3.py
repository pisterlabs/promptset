import openai
from gpt3.key import get_secret_key
from colorama import Fore, Style

openai.api_key = get_secret_key()


prompt = """Pick up the red container"""

prompt = """    This is a tweet sentiment classifier

    Tweet: "I loved the new Batman movie!"

    Sentiment: Positive

    ###

    Tweet: "I hate it when my phone battery dies."

    Sentiment: Negative

    ###

    Tweet: "My day has been 👍"

    Sentiment: Positive

    ###

    Tweet: "This is the link to the article"

    Sentiment: Neutral

    ###

    Tweet: "This new music video blew my mind"

    Sentiment:"""

prompt = """Select the best match of the 3 items:
{ball, boat, miniskirt}

Q: The child played with the toy.
A: ball

Q: The catcher threw it to first base.
A: ball

Q: The ship sunk.
A: boat

Q: The dress fluttered in the wind.
A: miniskirt

Q: Pick up the yellow bowl.
A: ball

Q: Pick up the bathtub toy.
A: boat

Q: Pick up the ship.
A:"""

prompt = """Complete the sentence from the stormlight archives by brandon sanderson:

Kaladin inhaled stomrmlight and began glowing. He lashed himself toward the sky, grabbing his spear. The Fused"""

prompt = """Mercury Venus Earth Mars Jupiter Saturn Uran"""

prompt = """Map these instructions into human language:
Move to home -> I am moving to the home position
###
Close Gripper -> Closing gripper now
###
Move over ball -> I am moving to the ball position
###
Close Gripper -> I am grasping the ball
###
Move over bucket -> I am getting ready to place the ball in the bucket
###
Open Gripper ->"""



response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=1, logprobs=100, n=2)
print(response)


for choice in response['choices']:
    print(f'{Fore.GREEN}{prompt}{Style.RESET_ALL}', end='')
    print(choice['text'])
    print('-'*20)
