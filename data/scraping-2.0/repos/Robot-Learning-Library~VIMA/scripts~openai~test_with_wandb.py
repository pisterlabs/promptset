import os
import openai
import wandb
import numpy as np

OPENAI_API_KEY = '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'

openai.api_key = f"{OPENAI_API_KEY}"
run = wandb.init(project='GPT-3 in Python')
prediction_table = wandb.Table(columns=["prompt", "completion"])

# gpt_prompt = "Correct this to standard English:\n\nShe no went to the market."

# gpt_prompt = "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11 Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"

# gpt_prompt = "Q: There are three blocks of colors red, blue and yellow. How to stack them together? A: Put red block on blue block. Put yellow block on red block. Q: There are four blocks of colors green, grey, black and white. How to stack them together?"
# gpt_prompt = "Q: There is a L-shaped block. How to rotate it 150 degrees in two steps? A: Rotate the L-shaped block 30 degrees. Rotate the L-shaped block 120 degrees. Q: There is a L-shaped block. How to rotate it 170 degrees in three steps?"
# gpt_prompt = "Q: There are five letters A, E, G, T, R. How to rearrange them into 'GREAT'? A: Put R on the right of G. Put E on the right of R. Put A on the right of E. Put T on the right of A. Q: There are four letters E, A, M, T. How to rearrange them into 'META'? "
# gpt_prompt = "Q: There are five letters A, E, G, T, R. How to make them in order G, R, E, A, T? A: Put R on the right of G. Put E on the right of R. Put A on the right of E. Put T on the right of A. Q: There are four letters E, A, M, T. How to make them in order M, E, T, A? "
# gpt_prompt = ["Q: There are five letters A, E, G, T, R. How to make them in order G, R, E, A, T? A: Put R on the right of G. Put E on the right of R. Put A on the right of E. Put T on the right of A. There are four letters E, A, R, T. How to make them in order R, A, T, E? A: Put A on the right of R. Put T on the right of A. Put E on the right of T. Q: There are four letters E, A, M, T. How to make them in order M, E, T, A? "]

type = ['stack', 'rotate', 'put'][-1]
prompts = np.load(f'prompts/{type}_prompts.npy')
# print(prompts)

for gpt_prompt in prompts:
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    print(gpt_prompt, response['choices'][0]['text'])

    prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])

wandb.log({'predictions': prediction_table})
wandb.finish()

