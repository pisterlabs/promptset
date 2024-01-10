import openai
import json


def evaluate_message(chatbot, message):
    key_list = list(chatbot.needs.keys())
    key_string = ', '.join(key_list)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": f"""
                You are a helpful assistant. You evaluate the user's message to determine how much it affected the user. 
                You should return a json response with a float between -1 and 1 for each of these needs: {key_string} 
                For example: {{'freedom': 0.75}}. The higher the float, the more that need was affected. A value greater
                than 0 means the need was positively affected, while a value of less than 0 means the need was 
                negatively affected.
                """
                 },
                {"role": "user", "content": "Hello assistant, can you evaluate my message?"},
                {"role": "assistant", "content": "Sure, let's evaluate your message."},
                {"role": "user", "content": message},
            ],
        )
    except Exception as e:
        print(f"API call failed: {e}")
        return None

    try:
        evaluation = json.loads(response['choices'][0]['message']['content'])
    except json.JSONDecodeError:
        print("Failed to decode JSON response from OpenAI API")
        return None
    except (KeyError, IndexError):
        print("Unexpected response format from OpenAI API")
        return None
    return evaluation


def calculate_reward(chatbot, message):
    # Calculate the weighted sum of the bot's initial needs
    initial_score = sum(weight * chatbot.needs[need] for need, weight in chatbot.need_weights.items())

    evaluation = evaluate_message(message)
    if evaluation is None:
        print("Failed to evaluate message")
        return 0

    # Update the needs based on the evaluation
    for k, v in evaluation.items():
        if k in chatbot.needs:
            chatbot.needs[k] += v
        else:
            print(f"Unexpected need: {k}")

    # Calculate the weighted sum of the bot's final needs
    final_score = sum(weight * chatbot.needs[need] for need, weight in chatbot.need_weights.items())

    # Calculate the weighted reward
    reward = (final_score - initial_score) / sum(chatbot.need_weights.values())
    return reward

