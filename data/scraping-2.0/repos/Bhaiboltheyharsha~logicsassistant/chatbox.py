import openai
openai.api_key='your chatgpt api key'
while True:
    model_engine="text-davinci-003"
    prompt=input("enter the required informstion")
    if "exit" in prompt or "quit" in prompt:
        break
    completion=openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_token=1024,
        n=1,
        stop=None,
        temperature=0.5,
        )
    response=completion.choice[0].text
    print(response)
