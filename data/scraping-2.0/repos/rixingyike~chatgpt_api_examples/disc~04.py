# disc/04.py
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
text_prompt = "请你扮演一个脱口秀演员，讲一个与‘艺述论’有关的笑话。"
completion = openai.Completion.create(
    model="text-davinci-003",
    prompt=text_prompt,
    temperature=0.5,
)

generated_text = completion.choices[0].text
print(generated_text)

# returns example text
# 一个男人把他的车停在艺术馆前，他走进去，看到一幅画，上面写着：“这是一幅艺述论。”他把头抬...