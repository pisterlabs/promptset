import openai

class GPT3():

    @staticmethod
    async def gpt3(stext):
        openai.api_key = 
        response = openai.Completion.create(
            model="davinci-instruct-beta",
            prompt=stext,
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        content = response.choices[0].text.split('.')
        # # print(content)
        return response['choices'][0]['text']
