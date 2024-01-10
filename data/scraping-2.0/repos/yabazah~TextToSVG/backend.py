# backend of the app

import openai

# new class for the backend
class backend_app:

    openai.api_key = 'YOUR API HERE'

    # function to generate SVG code from description given by user from the application.py file
    def generate_svg(self, userDesc, start, end):

        # # prompt for the GPT-3 engine
        # prompt = f"{description}\n\n{start}\n\n"

        # parameters for the GPT-3 engine
        response = openai.Completion.create(
            engine="davinci",
            prompt=userDesc + "\n\n" + start + "\n\n" + end,
            temperature=0.7,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.3
            # stop=["\n\n"]
        )

        # return the generated SVG code starting from the start tag and ending with the end tag 
        value = response.get("choices")[0]["text"]
        # split the value to start from the start tag and end with the end tag
        return value.split(start)[1].split(end)[0]
        # return response.choices[0].text.split(start)[1].split(end)[0]
        # return response.get("choices")[0]["text"]
    


