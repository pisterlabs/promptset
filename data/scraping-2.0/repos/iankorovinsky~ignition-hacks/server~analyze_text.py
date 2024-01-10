import openai
import os
import re

openai.api_key = "API_KEY"

def generate_text(question, text):
        print("generating gpt")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "user",
                "content": f"I am answering an interview question. The question asked is: {question}. My response was: {text}.\nTell me how I can improve my response based on the criteria of: professionalism, creativity, clarity, and originality. Then, provide a better response to the question that fits the criteria mentioned above. Start the overall response with a paragraph complementing me on my hard work."
                }
            ],
            temperature=0,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
        )
        print("received gpt")
        content = response.choices[0].message.content    
        print(content)
        print("returning gpt")
        return content


