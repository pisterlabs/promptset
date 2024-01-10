import openai
import os

class GPTParser:

    openai.api_key = os.environ.get('OPENAI_API_KEY')

    def data_extracter(self, response: str, delimiter: str):
        for i in range(len(response)):
            if response[i] == delimiter:
                for x in range(i+1, len(response)):
                    if response[x] == delimiter:
                        return response[i+1:x].lower()

    def datetime_parser(self, string: str):

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Carry out the task assigned. Do not provide an explanation."},
                {"role": "user", "content": "Transform the following date and time into the format DD-MM HH:MM (24 hour format) enclosed by % signs.\nUnformatted: " + string + "\nExample response: %15-06 17:43%\nFormatted: "},
            ]
        )

        #        response = openai.Completion.create(
        #            model="text-davinci-003",
        #            prompt="Transform the following date and time into the format DD-MM HH:MM (24 hour format) enclosed by % signs.\nUnformatted: " + string + "\nExample response: %15-06 17:43%\nFormatted: "
        #        )

        response = response.choices[0].message.content

        print(response)        
        
        # response = response.choices[0].text

        date = self.data_extracter(response, '%')

        return date

