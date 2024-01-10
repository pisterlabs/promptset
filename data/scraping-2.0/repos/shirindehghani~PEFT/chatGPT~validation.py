import openai

def validation_step(user_massage,system_massege, 
                model_name="ft:gpt-3.5-turbo-0613:university-of-zurich:chat-bot:8HVR8Mlb",
                temperature=0.8):
    completion = openai.ChatCompletion.create(
          model=model_name,
        messages=[
          {"role": "system","content": system_massege},
          {"role": "user","content": user_massage },
        {"role" : "assistant","content" : "Hello :), welcome to Coinfident, how can I assist you?"}], 
        temperature=temperature)
    return completion.choices[0]['message']['content']