import openai

def generate_koan(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Update the model to GPT-4
        messages=[
            {
                "role": "system",
                "content": "You are a wise Zen monk. When presented with a topic or question, craft a response as a Zen koan - an enigmatic parable that prompts contemplation without any overt or discernible lesson."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    koan = response['choices'][0]['message']['content']
    return koan

# Example Usage
if __name__ == "__main__":
    prompt = "What is the sound of one hand clapping?"
    koan = generate_koan(prompt)
    print(koan)
