# import openai

# # Set the API key
# openai.api_key = "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"

# # Define the prompt
# main_prompt = """This is a conversation with a sucide helpline operator."""

# while True:
#     prompt = input("You: ")
#     if prompt == "quit":
#         break

#     main_prompt += f"\nYou: {prompt} + \nOperator:"
#     # Generate text
#     completions = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=main_prompt,
#         temperature=0.9,
#         max_tokens=150,
#         top_p=1,
#         frequency_penalty=0.0,
#         presence_penalty=0.6,
#         stop=[" You:", " Operator:"]
#     )

#     # Print the generated text
#     if len(completions.choices) == 0:
#         print("Operator: I'm sorry, I don't understand.")
#         continue
    
#     response = completions.choices[0].text
#     print("Operator: " + response)





def hello_world(request):
    import openai
    openai.api_key = "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"

    request_json = request.get_json()

    if request_json['path'] == "help":
      helprequest(request_json['input'])

   

def helprequest(prompt):
    import openai

    main_prompt = """This is a conversation with a sucide helpline operator."""
    main_prompt += f"\nYou: {prompt} + \nOperator:"
    # Generate text
    completions = openai.Completion.create(
        model="text-davinci-003",
        prompt=main_prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" You:", " Operator:"]
    )

    if len(completions.choices) == 0:
        return "Operator: I'm sorry, I don't understand.", 400

    response = completions.choices[0].text
    return response

{"path": "help", "input": "Hello"}