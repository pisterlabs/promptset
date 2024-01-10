from chaincrafter import Chain, Prompt
from chaincrafter.models import OpenAiChat


gpt3_chat = OpenAiChat(0.7, "gpt-3.5-turbo")
gpt4_chat = OpenAiChat(0.7, "gpt-4")

system_prompt = Prompt("You are a principal software engineer with expertise in JavaScript and Python")
code_to_fix_prompt = Prompt("""The following is source code for a Python function

```python
{code}
```

List suggestions for improvements in the following format:
1. Suggestion 1
2. Suggestion 2
""", code=str)
code_and_suggestions_prompt = Prompt("""The following is source code for a Python function

```python
{code}
```

Apply the following suggestions to the code:
{code_suggestions}
""", code=str, code_suggestions=str)

suggestions_chain = Chain(system_prompt, (code_to_fix_prompt, "output"))
apply_suggestions_chain = Chain(system_prompt, (code_and_suggestions_prompt, "output"))
input_vars = {
    "code": """def complete(self, messages):
    import openai
    completion = openai.ChatCompletion.create(
        model=self._model_name,
        temperature=self._temperature,
        messages=messages,
    )
    self.usage["prompt_tokens"] += completion.usage["prompt_tokens"]
    self.usage["completion_tokens"] += completion.usage["completion_tokens"]
    self.usage["total_tokens"] += completion.usage["total_tokens"]
    return completion.choices[0].message["content"]
"""
}
print("GPT-3.5 Turbo: running chain 'suggestions_chain'")
messages_gpt3 = suggestions_chain.run(gpt3_chat, input_vars)
print("GPT-3.5 Turbo: results for 'suggestions_chain'")
for message in messages_gpt3:
    print(f"{message['role']}: {message['content']}")

print("GPT-4: running chain 'suggestions_chain'")
messages_gpt4 = suggestions_chain.run(gpt4_chat, input_vars)
print("GPT-4: results for 'suggestions_chain'")
for message in messages_gpt4:
    print(f"{message['role']}: {message['content']}")
print('#######################')
print("GPT-3.5 Turbo: running chain 'apply_suggestions_chain' based on GPT-4 suggestions")
messages_gpt3_applied = apply_suggestions_chain.run(gpt3_chat, { "code": input_vars["code"], "code_suggestions": messages_gpt4[-1]["content"] })
print("GPT-3.5: results for 'apply_suggestions_chain'")
for message in messages_gpt3_applied:
    print(f"{message['role']}: {message['content']}")
print("GPT-4: running chain 'apply_suggestions_chain' based on GPT-3.5 Turbo suggestions")
messages_gpt4_applied = apply_suggestions_chain.run(gpt4_chat, { "code": input_vars["code"], "code_suggestions": messages_gpt3[-1]["content"] })
print("GPT-4: results for 'apply_suggestions_chain'")
for message in messages_gpt4_applied:
    print(f"{message['role']}: {message['content']}")
