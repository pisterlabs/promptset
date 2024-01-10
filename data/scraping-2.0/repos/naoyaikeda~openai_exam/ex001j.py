# %%
!source setkey.sh

# %%
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.Completion.create(
  engine="davinci",
  prompt="人類は多くの問題を抱えていた。大量の破壊兵器・増えつづけた人口・国際的なテロ・国家間の極端な貧富の差･･･これら問題を解決する為にある計画が実現に向かう。 地球上すべての国家をある1つのコンピュータによって統括しようという大胆な計画。そしてその中央処理装置はMESIAと呼ばれていた。",
  temperature=0.7,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# %%
print(response)


# %%
response_text = response["choices"][0]["text"]


# %%
response_text

# %%
