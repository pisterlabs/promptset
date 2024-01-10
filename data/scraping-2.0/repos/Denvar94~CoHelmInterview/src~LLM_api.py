from openai import OpenAI

class OpenAILInker:

    def __init__(self):
        self.client = OpenAI()

    def pass_messages(self, messages):
        c = self.client.chat.completions.create(
            messages=messages,
            model="gpt-4",
        )

        content = c.choices[0].message.content

        return content

    def pass_message(self, sys, message):
        c = self.client.chat.completions.create(
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": message}],
            model="gpt-4",
        )

        content = c.choices[0].message.content

        return content

    def call_api(self, sys, message, model="gpt-3.5-turbo"):
        msgs = [{"role": "system", "content": sys},
                      {"role": "user", "content": message}]
        try:
            return self.client.chat.completions.create(model=model, messages=msgs)
        except openai.error.RateLimitError as e:
            retry_after = int(e.headers.get("retry-after", 60))
            print(f"Rate limit exceeded, waiting for {retry_after} seconds...")
            time.sleep(retry_after)
            return call_api(params, model=model)
