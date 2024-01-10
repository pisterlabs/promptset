#%%
from k import k
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = k

def open_ai_chatpgt(prompt, context):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=k,
    )

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""

                Prompt: {prompt}

                Context: {context}

                """,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content
# %%

open_ai_chatpgt("What is machine larning?", "remember the difference between AI and ML")
# %%
