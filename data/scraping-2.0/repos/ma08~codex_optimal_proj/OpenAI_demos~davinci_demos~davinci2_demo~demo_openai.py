import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-002",
  prompt="\"\"\"\nSort list with a time complexity of O(n^logn)\n\"\"\"",
  temperature=0,
  max_tokens=300,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response)
print()
print()
print()

for choice in response["choices"]:
    finish_reason = choice["finish_reason"]
    print(f"REASON {finish_reason}")
    if(finish_reason == "stop"):
        with open("demo_out.py", "w") as fp:
            fp.write(choice["text"])


# if __name__=="__main__":


# print(response["id"])