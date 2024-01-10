import os
import openai
import random
import main

openai.api_key = "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"

def ai(input):

    text=f"{input[8:30]} is the prompt and response :: \n***************************************\n"

    response = openai.Completion.create(
    model="text-curie-001",
    prompt= input,
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    try:
        print(response["choices"][0]["text"])
        text+=response["choices"][0]["text"]
        if not os.path.exists("openai_files"):
            os.mkdir("openai_files")

        with open(f"openai_files/topic-{input[8:30]},","w") as f:
            f.write(text) 
    except Exception as e:
        print(f"{e}")
    

'''
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "We Can End Gun Violence\n\nThis petition is for the United States Congress to make commonsense, public safety measures a top priority and pass gun violence legislation that will help prevent tragedies like the shooting at Marjory Stoneman Douglas High School in Parkland, Florida.\n\n\n\nIn the past two years, 43 people have been killed with guns in the United States. This number is simply unacceptable and must be changed. We can end gun violence by passing commonsense gun violence prevention legislation. This includes but is not limited to:\n\n-Prohibiting people who are not legally allowed to own a gun from buying one\n\n-Requiring a background check for all gun purchases\n\n-Making it illegal to sell guns to people who have been convicted of a violent felony\n\n-Strengthening our mental health system so that people with mental illness cannot purchase guns\n\n-Making it easier for authorities to confiscate guns from people who are deemed a danger to themselves or others\n\nWe demand that our elected officials take action to end gun violence and make our communities safer. Please sign this petition and call on your elected officials to pass commonsense gun violence prevention legislation."
    }
  ],
  "created": 1689504116,
  "id": "cmpl-7ctUKTeotJhNo9gpa5qVCQ4NHVoox",
  "model": "text-curie-001",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 239,
    "prompt_tokens": 1,
    "total_tokens": 240
  }
}
'''