# Example based on the prompts from MiniChain:
# - https://github.com/srush/MiniChain/blob/main/examples/math_demo.py
# - https://github.com/srush/MiniChain/blob/main/examples/math.pmpt.tpl
from chaincrafter import Chain, Prompt
from chaincrafter.models import OpenAiChat

math_questions_and_code = [
    (
        "What is 37593 * 67?",
        "print(37593 * 67)",
    ),
    (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "print((16-3-4)*2)",
    ),
    (
        "How many of the integers between 0 and 99 inclusive are divisible by 8?",
        """count = 0
for i in range(0, 99+1):
    if i % 8 == 0: count += 1
print(count)"""
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "print(2 + 2/2)",
    ),
]
math_prompt_text = ""
for question, code in math_questions_and_code:
    math_prompt_text += f"""#### Question:
* {question}

#### Code:

```python
{code}
```
"""
math_prompt_text += """#### Question:

* {question}

#### Code:
"""

chat_model = OpenAiChat(temperature=0.7, model_name="gpt-3.5-turbo")
system_prompt = Prompt("You are a helpful assistant")
math_prompt = Prompt(math_prompt_text, question=str)
chain = Chain(
    system_prompt,
    (math_prompt, "code"),
)
questions = [
    "What is the sum of the powers of 3 (3^i) that are smaller than 100?",
    "What is the sum of the 10 first positive integers?",
    "Carla is downloading a 200 GB file. She can download 2 GB/minute, but 40% of the way through the download, the download fails. Then Carla has to restart the download from the beginning. How load did it take her to download the file in minutes?"
]
for question in questions:
    messages = chain.run(chat_model, {"question": question})
    print(f"Question: {question}")
    content = messages[-1]["content"]
    code = content[content.find("```python") + len("```python"):content.rfind("```")]
    print(f"Code: {code}")
    exec(code)
    print()
