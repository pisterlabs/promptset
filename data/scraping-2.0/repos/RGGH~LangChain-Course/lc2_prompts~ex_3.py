import openai


class Chatty:
    def chat(self, input):
        messages = [
            {"role": "system", "content": "You are a helpful,\
                upbeat and funny assistant"},
            {"role": "user", "content": input},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )

        parsed_response = self.parse(response)
        return parsed_response

    def parse(self, response):
        content = response.choices[0].message["content"]
        return content


m = Chatty()
output = m.chat("why did the chicken cross the road?")
print(output)
