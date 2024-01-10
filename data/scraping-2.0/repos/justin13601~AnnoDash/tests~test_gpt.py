import os
import time
import openai


def main(system_prompt, user_message):
    start_time = time.time()
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = 'gpt-3.5-turbo'

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0,
        max_tokens=2000,
    )

    print(response)
    elapsed_time = time.time() - start_time
    print('--------GPT Ranking Time:', elapsed_time, 'seconds--------')
    return


if __name__ == "__main__":
    system_prompt = """
    
    """

    user_message = """
    
    """

    main(system_prompt, user_message)
