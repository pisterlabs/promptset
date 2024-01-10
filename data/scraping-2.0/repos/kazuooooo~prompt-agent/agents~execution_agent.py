import openai
import os
from colorama import Fore, Style

def execute(
  prompt: str
) -> str:
  """
  Execute a prompt and return the output

  Paramters:
    propmt: str
  
  Returns:
    str: output of the fixed prompt
  """

  print(Fore.RED + "*****実行エージェント*****")
  print("プロンプト:", prompt)

  response = openai.ChatCompletion.create( #type: ignore
    model=os.environ.get("LLM_MODEL"),
    messages=[{"role": "system", "content": prompt }],
    temperature=0
  )
  output = response["choices"][0]["message"].content #type: ignore
  print("↓\n出力:", output) #type: ignore
  print(Style.RESET_ALL)
  return output #type: ignore