import openai

# -------------------------------------------------------------------

# Just Talk Block


# Set your OpenAI API key here
api_key = "YOUR_OPENAI_API_KEY"

# Initialize the OpenAI API client
openai.api_key = api_key

def generate_response(prompt, model_name="gpt-4"):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.InvalidRequestError:
        if model_name != "gpt-3.5-turbo":
            print("Failed to access GPT-4, falling back to GPT-3.5-turbo.")
            return generate_response(prompt, "gpt-3.5-turbo")
        else:
            raise


while True:
    # Get user input
    input_text = input("Enter your input text (or type 'exit' to quit): ")

    if input_text.lower() == "exit":
        print("Goodbye!")
        break

    # Generate a response
    generated_response = generate_response(input_text)

    # Print the generated response
    print("Generated Response:")
    print(generated_response)


    
    
    
    
    
    
    # -------------------------------------------------------------------------
    
    # Talk With Fixed Prompt Block

# # Set your OpenAI API key here
# api_key = "YOUR_OPENAI_API_KEY"

# # Initialize the OpenAI API client
# openai.api_key = api_key

# # Fixed prompt that will be added to the beginning of each input
# fixed_prompt = "You are acting as a Canadian fratboy. Please phrase your response as such using 'Bro' and 'eh' frequently.\n"

# def generate_response(prompt, model_name="gpt-4"):
#     try:
#         response = openai.ChatCompletion.create(
#             model=model_name,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except openai.error.InvalidRequestError:
#         if model_name != "gpt-3.5-turbo":
#             print("Failed to access GPT-4, falling back to GPT-3.5-turbo.")
#             return generate_response(prompt, "gpt-3.5-turbo")
#         else:
#             raise





# while True:
#     # Get user input
#     input_text = input("Enter your input text (or type 'exit' to quit): ")

#     if input_text.lower() == "exit":
#         print("Goodbye!")
#         break

#     # Generate a response
#     generated_response = generate_response(input_text)

#     # Print the generated response
#     print("Generated Response:")
#     print(generated_response)
