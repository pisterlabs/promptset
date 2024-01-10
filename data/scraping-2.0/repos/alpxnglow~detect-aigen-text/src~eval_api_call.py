import openai
import pandas as pd
import time

MODEL_ENGINE = "gpt-3.5-turbo" 
EDITED_PROMPTS_FILE = 'data/edited_prompts.txt'
AI_TEST_ESSAYS_OUTPUT_FILE = 'data/ai_test_essays_12282023180300.csv'

edited_prompt_input = open(EDITED_PROMPTS_FILE, "r")
prompts = edited_prompt_input.readlines()

# Create a dummy dataframe
df = pd.DataFrame(columns=['essay', 'generated'])
num = 0

for i in range(1000):
  response = openai.chat.completions.create(
    model=MODEL_ENGINE,
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": prompts[i]}
    ]
  )
  df.loc[len(df.index)] = [response.choices[0].message.content, 1]
  print(df.loc[len(df.index)-1])
  df.to_csv(AI_TEST_ESSAYS_OUTPUT_FILE, encoding='utf-8', index=False)
  num += 1
  if num == 3:
    time.sleep(60) # sleep for a minute due to API call rate limit
    num = 0

#close files
edited_prompt_input.close()
#ai_test_essays_output.close()
