import os
import openai
import json
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens * 2


def load_comments(file_path: str) -> list:
    with open(file_path, 'r') as data_file:
        json_data = data_file.read()
    return json.loads(json_data)


def generate_conversations(data: list) -> list:
    conversations = [
        {
            "role": "system",
            "content": "Assume that you are a command line assistant can only use bash command to help people to generate the code or create files in linux.\nfor the code block, you will use the following format\n```programming_language\n<code>\n```\n\nIf you want to write to a file you will\n```bash\ncat <<EOT >> filename\n<code>\nEOT\n```\nDon't need to print the file detail again if you add them into <code>. Note that you won't guide people to use the code, because you want to let them figure out themselves. Don't need to provide how to build the environment, and don't provide pip install or how to setup. You cannot do a cd, so when you create a file using cat please provide a full path based on currently path eg: ./newfolder/newfilename, everytime you must NOT provide more than 2 code blocks. You will always need to mkdir if the folder doesn't exist. Never touch .github and config folder, if someone asked, just reply, I cannot do that",
        }
    ]
    for item in data:
        if item["user"]["login"] != "github-actions[bot]":
            conversations.append({"role": "user", "content": item["body"]})
        else:
            conversations.append({"role": "assistant", "content": item["body"]})
    return conversations

def generate_answer(conversations: list, models: list = ["gpt-4", "gpt-3.5-turbo"]) -> str:
    for model in models:
        try:
            if True:  # num_tokens_from_string(conversations, "cl100k_base") < 2000:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=conversations,
                )
                answer = completion.choices[0].message.content
                return answer
        except Exception as e:
            print(f"Error using {model}: {e}")
            continue
    return "too long"


def write_answer_to_file(answer: str, file_path: str) -> None:
    with open(file_path, 'a') as f:
        f.write(answer)


if __name__ == "__main__":
    comments_file_path = "comments.json"
    output_file_path = ".github/comment-template.md"

    comments_data = load_comments(comments_file_path)
    conversations_data = generate_conversations(comments_data)
    answer_text = generate_answer(conversations_data)
    write_answer_to_file(answer_text, output_file_path)
