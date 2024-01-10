import numpy as np
from scipy.special import logsumexp
from datasets import load_dataset
import openai
import tiktoken

from logit_estimation.estimators import naive_estimate, GptSampler, gptdiffsearch

#logits = np.load("saved_logits-gpt/0-diff-*-eps1e-06.npy")
buggy_logits = np.load("saved_logits-gpt/0-diff-3457440-eps1e-06.npy")
# highest should be 4815 for example 0, but this was negated in the buggy code
logits = -(buggy_logits - buggy_logits[4815])
logits = logits - logsumexp(logits)


dataset = load_dataset("wentingzhao/one-million-instructions")["train"]
d = dataset[0]
#prefix = f"[INST] <<SYS>>\n{d['system']}\n<</SYS>>\n {d['user']} [/INST]"
#prefix = d["system"] + "\n\n" + d["user"]
prefix = d["user"]

model = "gpt-3.5-turbo-instruct"
#model = "gpt-3.5-turbo"

k = 10
topk = np.argpartition(logits, -k)[-k:]
enc = tiktoken.encoding_for_model(model)
print(enc.decode(topk))

response = openai.Completion.create(
    model = model,
    prompt=prefix,
    temperature=1,
    max_tokens=1,
    logprobs=5,
)

top5_tokens = list(response.choices[0].logprobs.top_logprobs[0].keys())
top5_ids = [enc.encode(x) for x in top5_tokens]
import pdb; pdb.set_trace()
