# import tkinter as tk
#
# import py_tool
# import screenshot
#
# scale = py_tool.get_screen_scale_rate()
# py_tool.eliminate_scaling_interference()
# top = tk.Tk()
#
# screenshot.Screenshot(top, scale)
#
# top.mainloop()
#
# import openai
#
# # Set your API key
# openai.api_key = "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"
# # Use the GPT-3 model
# completion = openai.Completion.create(
#     engine="text-davinci-002",
#     prompt="Once upon a time, in a land far, far away, there was a princess who...",
#     max_tokens=1024,
#     temperature=0.5
# )
# # Print the generated text
# print(completion.choices[0].text)

import openai

openai.api_key = "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"  # 这里是你的api-key


def askChatGPT(question):
    prompt = question
    model_engine = "text-davinci-003"

    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    print(message)


askChatGPT("what is love")

