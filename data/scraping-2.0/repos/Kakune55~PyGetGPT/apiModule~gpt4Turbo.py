import openai , config

openai.api_key = config.readConf()["gpt4.0turbo"]["Authorization"]
openai.base_url = config.readConf()["gpt4.0turbo"]["url"]

def service(prompt,history = ""):
    if history == "":
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
    else:
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "user", "content": history[1]["user"]},
                {"role": "assistant", "content": history[1]["bot"]},
                {"role": "user", "content": history[0]["user"]},
                {"role": "assistant", "content": history[0]["bot"]},
                {"role": "user", "content": prompt},
            ]
        )
    if response.choices[0].finish_reason == "stop":
        return 200, response.choices[0].message.content, int(response.usage.total_tokens*45) #45倍tokens消耗
    else:
        return 50 , "API Error!", 0