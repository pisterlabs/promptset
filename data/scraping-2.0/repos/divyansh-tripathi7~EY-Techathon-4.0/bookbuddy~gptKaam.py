import openai


class GPT_3:
    def __init__(self, api_key):
        openai.api_key = api_key

        self.completion = openai.Completion
        self.options = {
            'engine': 'text-davinci-002',
            'temperature': 0.25,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'max_tokens': 512
        }

    def __call__(self, prompt, options=None):
        return self.prediction(prompt, options)

    def prediction(self, prompt, options=None):
        if not options:
            options = self.options

        return self.completion.create(prompt=prompt, **options)['choices'][0]['text']

    def teach(self, text):
        prompt = f'explain the following topic to a UG student in a conventional way.\n\n Topic: {text}'
        return self.prediction(prompt=prompt)

    def script(self, text):
        prompt = f'Provide script to create a short and crisp video to explain about : {text}'
        return self.prediction(prompt=prompt)

    def resources2(self, text  ):
        prompt = f'Tell me resources to learn about {text} in the form of a list with numbering '
        return self.prediction(prompt=prompt)

    def summarize(self, text, num=None):
        prompt = f'Summarize this yt video script to explain what all is covered in the video in about {num} words. \n\n script: {text}'

        return self.prediction(prompt=prompt)

    def summarize2(self, text, num=None):
        prompt = f'Summarize pdf to explain what all is covered in the book \n\n book: {text}'

        return self.prediction(prompt=prompt)

    def topics(self, text):
        prompt = f'Mention all the topics included in this piece of text !\n\n text: {text}'
        # time.sleep(1)
        return self.prediction(prompt=prompt)

    def translate(self, text, l):
        prompt = f'Translate provided text into the {l} language !\n\n text: {text}'
        # time.sleep(1)
        return self.prediction(prompt=prompt)

    # in the form of a dictionary like ["question": "Question 1","options": ["Option 1", "Option 2", "Option 3", "Option 4"], "correct_answer": "Option 1" ]" and return a dictionary not a string

    def QuizMe(self, text):
        prompt = f'please make a MCQ quiz of 5 question related to the following text with correct options in the end !\n\n{text}'
        # time.sleep(1)
        return self.prediction(prompt=prompt)
    
    def qna(self , question , context):
        prompt = f'The question is {question} answer this on the basis of this text: {context}'
        return self.prediction(prompt = prompt)
