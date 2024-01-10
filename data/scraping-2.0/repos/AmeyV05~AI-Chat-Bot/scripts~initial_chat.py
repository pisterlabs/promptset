#%%
import openai
# import readline


#%%
def generate_response(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.2,
    )
    return response.choices[0].text.strip()

def main():
    history = []
    while True:
        user_input = input("You: ")
        history.append(user_input)
        prompt = "\n".join(history)
        response = generate_response(prompt)
        history.append(response)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
