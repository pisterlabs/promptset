
import openai

class ResponseGenerator:
    def __init__(self):
        self.api_key = "your_openai_api_key"
        openai.api_key = self.api_key

    def generate_responses(self, emails):
        responses = []
        for email in emails:
            prompt = f"The email says: {email['body']}\n\nResponse:"
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=150
            )
            responses.append({
                'to': email['from'],
                'subject': f"Re: {email['subject']}",
                'body': response.choices[0].text.strip()
            })
        return responses
