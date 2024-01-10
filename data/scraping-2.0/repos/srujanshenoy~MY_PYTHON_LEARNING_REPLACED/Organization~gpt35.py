import openai
#
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",

    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

# def generate_response(prompt):
#     completions = openai.Completion.create(
#         engine = "gpt-3.5-turbo",
#         prompt = prompt,
#         max_tokens = 1024,
#         n = 1,
#         stop = None,
#         temperature = 0
#     )
#     return completions
# 
# while True:
#     prompt = input(">>>")
#     print(generate_response(prompt))
#
