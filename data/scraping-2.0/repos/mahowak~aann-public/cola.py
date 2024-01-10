import ssl
import pandas as pd
import openai
import numpy as np
import sklearn

KEY = 'INSERT OPEN AI KEY HERE'
ssl._create_default_https_context = ssl._create_unverified_context

def getcolaprompt(x):
  return f"""Now we are going to say which sentences are acceptable (i.e., grammatical) and which are not.

  Sentence: Flosa has often seen Marn.
  Answer: good

  Sentence: Chardon sees often Kuru.
  Answer: bad

  Sentence: Bob walk.
  Answer: bad

  Sentence: Malevolent floral candy is delicious.
  Answer: good

  Sentence: The bone chewed the dog.
  Answer: good

  Sentence: The bone dog the chewed.
  Answer: bad

  Sentence: I wonder you ate how much.
  Answer: bad

  Sentence: The fragrant orangutan sings loudest at Easter.
  Answer: good

  Sentence: {x}
  Answer:"""


def get_judgment(prompt, model):
    openai.api_key = KEY
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0,
        max_tokens=1,
        echo=True,
        top_p=1,
        logprobs=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return (response["choices"][0]["logprobs"]["tokens"][-1],
            np.exp(response["choices"][0]["logprobs"]["token_logprobs"][-1]))

def run_cola_test():
    """Test the COLA data on the prompt."""
    cola_dev = pd.read_csv("http://archive.nyu.edu/bitstream/2451/60441/10/in_domain_dev.txt", sep="\t",
                           names=["source", "accept", "star", "sent"])

    judgments = [get_judgment(getcolaprompt(i)) for i in list(cola_dev.sent)]
    cola_dev["gpt3judgments"] = judgments
    cola_dev["gpt3judgment_binary"] = ["good" in i[0]
                                    for i in cola_dev["gpt3judgments"]]
    cola_dev["correct"] = cola_dev["gpt3judgment_binary"] == cola_dev["accept"]
    print("COLA accuracy", cola_dev["correct"].mean())
    print(sklearn.metrics.matthews_corrcoef(
        cola_dev["gpt3judgment_binary"], cola_dev["accept"]))


if __name__ == "__main__":
    run_cola_test()