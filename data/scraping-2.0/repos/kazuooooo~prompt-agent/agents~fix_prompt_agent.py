import openai
import json
import os
from agents.helpers.list_to_bullet import list_to_bullet
from colorama import Fore, Style

def fix_prompt(
  current_prompt: str,
  improvements: list[str],
) -> str:
  """
  Fix prompt based on improvements.

  Paramters:
    current_prompt(str)
    improvements(list): list of improvements suggested by evaluation_agent
  
  Returns:
    str: Fixed prompt
  """

  agent_prompt: str = f"""
あなたはプロンプトエンジニアです。
このプロンプトを改善点に基づいて、修正してください。

プロンプト
```
{current_prompt}
```

改善点
```
{list_to_bullet(improvements)}
```
  """

  print(Fore.LIGHTBLUE_EX + "*****プロンプト修正エージェント*****")
  print("現在のプロンプト:", current_prompt)
  print(f"""改善点:
{list_to_bullet(improvements)}""")

  response = openai.ChatCompletion.create( #type: ignore
    model=os.environ.get("LLM_MODEL"),
    messages=[{"role": "system", "content": agent_prompt }],
    functions=[
      {
        "name": "fix_prompt",
        "description": "現在のプロンプトを改善点に基づいて修正する",
        "parameters": {
          "type": "object",
          "properties": {
            "fixed_prompt": {
              "type": "string"
            }
          }
        }
      }
    ],
    function_call={ "name": "fix_prompt"},
    temperature=0
  )

  response_message = response["choices"][0]["message"] #type: ignore
  function_args = json.loads(response_message["function_call"]["arguments"]) #type: ignore
  fixed_prompt = function_args['fixed_prompt']
  print("↓\n修正されたプロンプト:", fixed_prompt, "\n")
  print(Style.RESET_ALL)

  return fixed_prompt