# try to generate a script in a single prompt, exported as a json

import openai
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

openai.api_key = openai_API_key

script = "Arsenal Football Club is a professional football club based in London, England. The club was founded in 1886 as Dial Square Football Club and in 1893 became known as Arsenal. They play their home matches at the Emirates Stadium and are one of the most successful clubs in English football history, having won 13 league titles and 14 FA Cups. Arsenal has a loyal fan base and is known for playing attacking and stylish football."

response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": f"Generate a quiz with 2 questions, each with 5 choices based on the following extract from a lecture: {script}. The quiz must be exported in the json format using the following fields: quiz_name, question_count, questions, prompt, answer, options"}
    ]
)
answer = response["choices"][0]["message"]["content"]

print(answer)
print(type(answer))
