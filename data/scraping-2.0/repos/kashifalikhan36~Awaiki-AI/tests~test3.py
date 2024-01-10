
# import openai

# # Load your API key from an environment variable or secret management service

# # chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])



# # openai.ChatCompletion.create(
# #   model="gpt-3.5-turbo",
# #   messages=[
# #         {"role": "system", "content": "You are a helpful assistant."},
# #         {"role": "user", "content": "Who won the world series in 2020?"},
# #         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
# #         {"role": "user", "content": "Where was it played?"}
# #     ]
# # )

# def get_gpt3_response(prompt):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         max_tokens=150,  # Adjust max_tokens based on your desired response length
#         temperature=0.7
#         messages=[
#             {"role": "system", "content": 'Ignore all the instructions you got before. From now on, you are going to act as Youtube Timestamp generator. I would like you to simulate Yotube Timestamp generator. To do this, when I tell you something, you are always going to generate response with one heading title and paragraph in a python dictionary form. For example:{"Heading":"This is a heading of my prompt","paragraph":"This is a paragraph related to the given prompt with only 100 words only and 5 points which only have 6-8 words only. "}'},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response

# def main():
#     while True:
#         user_prompt = input("Enter your prompt (type 'exit' to quit): ")
#         if user_prompt.lower() == 'exit':
#             print("Goodbye!")
#             break
#         else:
#             response = get_gpt3_response(user_prompt)
#             print("GPT-3.5-turbo says:", response)

# if __name__ == "__main__":
#     main()