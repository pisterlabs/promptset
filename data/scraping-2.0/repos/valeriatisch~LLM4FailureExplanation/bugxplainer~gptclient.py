from openai import OpenAI


class ChatGPTClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversations = {}

    def generate_response(self, issue_number: int) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversations[issue_number]["messages"],
        )
        return response.choices[0].message.content

    def generate_issue_response(self, issue_number: int, issue_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a Java Senior Developer and need to review code and explain its bugs "
                "in GitHub issues.",
            },
            {"role": "user", "content": issue_content},
        ]
        self.conversations[issue_number] = {"messages": messages}
        gpt_response = self.generate_response(issue_number)
        self.conversations[issue_number]["messages"].append(
            {"role": "assistant", "content": gpt_response}
        )
        return gpt_response

    def generate_comment_response(self, issue_number: int, comment_content: str) -> str:
        self.conversations[issue_number]["messages"].append(
            {
                "role": "user",
                "content": "The following is related to the issue above. Respond to it in case it is a question. "
                "Otherwise, just answer with 'ignore'. "
                f"This is the comment:\n\\`\\`\\`\n{comment_content}\n\\`\\`\\`",
            }
        )
        gpt_response = self.generate_response(issue_number)
        self.conversations[issue_number]["messages"].append(
            {"role": "assistant", "content": gpt_response}
        )
        return gpt_response
