import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = """
##### Translate this code from Haskell to C++
### Haskell

data Node a = Node2 a a | Node3 a a a
data FingerTree a = Empty | Single a | Deep (Digit a) (FingerTree (Node a)) (Digit a)
type Digit a = [a]

### C++
"""

response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  temperature=0,
  max_tokens=128,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["###"]
)

print(response)
