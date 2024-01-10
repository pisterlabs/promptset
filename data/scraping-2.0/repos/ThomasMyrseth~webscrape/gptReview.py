import openai
openai.api_key = "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"
conversation_history = [{"role": "system", "content": "You are a financial data analacyst. Summarise the data ypu are given and point out important changes compared to previous reports"}]


class gptReview:
    def __init__(self):
        None
        
    def askGpt(self, prompt):
        message = {"role": "user", "content": prompt}
        conversation_history.append(message)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )

        generatedText = response.choices[0].message["content"]
        print("askGpt ran succesfully")
        return generatedText