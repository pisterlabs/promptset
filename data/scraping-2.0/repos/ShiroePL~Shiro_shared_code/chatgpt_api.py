# Note: you need to be using OpenAI Python v0.27.0 for the code below to work

import openai
def send_to_openai(messages):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature = 0.6
        )

        answer = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens
        
        return answer, prompt_tokens, completion_tokens, total_tokens

    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        return f"Error: {e.args[0]}", 0, 0, 0

    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        return "OpenAI API is currently overloaded", 0, 0, 0

    except openai.error.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        if 'timed out' in str(e):
            return "Request to OpenAI API timed out", 0, 0, 0
        else:
            return f"Error: {e.args[0]}", 0, 0, 0


#   #completion_tokens = result[2] or _,_,completion_tokens,_ = result if i would like ot take only one


if __name__ == "__main__":
    # answer, prompt_tokens, completion_tokens, total_tokens = send_to_openai()
    # print("answer: " + answer)
    # print("prompt_tokens: " + str(prompt_tokens))
    # print("completion_tokens: " + str(completion_tokens))
    # print("total_tokens: " + str(total_tokens))
    pass
    
