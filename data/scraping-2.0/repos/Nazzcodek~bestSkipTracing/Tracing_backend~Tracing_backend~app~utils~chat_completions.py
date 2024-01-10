import requests
import json
from transformers import GPT2Tokenizer


from config import OPENAI_API_KEY, OPENAI_ORG_ID


def summarize_conversation(conversation):
    prompt = "Summarize the conversation:\n"
    for message in conversation:
        prompt += f"{message['role']}: {message['content']}\n"
    response = get_response(prompt)
    return response


def count_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    print("token_count_________",token_count)
    return token_count


max_tokens_limit = 4096


def get_response(previous_chats, current_chat):
    try:

        print(current_chat['content'])

        url = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": []
        }
        # , ensure_ascii = False).encode('utf-8')

        headers = {
            'OpenAI-Organization': OPENAI_ORG_ID,
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + OPENAI_API_KEY
        }

        conversation = []
        total_tokens = sum(len(message['content'].split()) for message in previous_chats)
        print(total_tokens)
        if total_tokens > max_tokens_limit:
            print(total_tokens)
            summarized_conversation = summarize_conversation(previous_chats)
            conversation = [{'role': 'system', 'content': summarized_conversation}]
        else:
            conversation = previous_chats

        # current_message = {'role': 'user', 'content': 'What are the odds for the next match?'}
        combined_tokens = total_tokens + count_tokens(current_chat['content'])
        # combined_tokens = total_tokens + len(current_chat['content'].split())
        print("combined_tokens", combined_tokens)
        if combined_tokens <= max_tokens_limit:
            conversation.append(current_chat)

        else:
            return {
                "success": False,
                "error": "Unable to process your message, please reset the chat or reduce the size of the input message"
            }

        payload["messages"] = conversation
        payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        print(payload)
        response = requests.request("POST", url, headers=headers, data=payload)

        # print(response)
        print(response.status_code)

        if response.status_code != 200:
            return {
                "success": False,
                "error": "Unable to process chat"
            }
        print(payload)

        # print(headers)

        print(response.json())
        response = response.json()
        # print("response",response)
        # print("response choices",response)
        # Process the response
        if response and "choices" in response:
            print("_________")
            reply = response["choices"][0]["message"]
            # print("Assistant's reply:", reply)
            return {
                "success": True,
                "data": reply
            }

        else:
            print("Request failed with status code:", response["status"])
            print("Error message:", response["message"])

    except Exception as e:
        print("error", e)
        return False
