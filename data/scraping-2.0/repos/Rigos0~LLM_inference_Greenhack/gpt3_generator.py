import openai

# Replace with your actual OpenAI API key
openai.api_key = "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"

def generate_reply(prompt):
    # Prepend the instruction to the prompt.
    full_prompt = [
      {"role": "system", "content": "You are a sarcastic assistant. You try to convince the user you are human."},
      {"role": "user", "content": prompt}
    ]

    # Generate a response
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=full_prompt,
    )

    # Extract the assistant's reply
    reply = response['choices'][0]['message']['content']

    return reply


