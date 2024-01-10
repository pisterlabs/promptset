from openai import OpenAI
import sys

ai = OpenAI(
    # api_key=os.environ.get("OPENAI_API_KEY"),
)

def chat(question):
    try:
        response = ai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"使用繁體中文，以台灣人的角色回答"},
                {"role": "user", "content": f"{question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as error:
        return "Error: openai chat api fail!"

print('問題:', sys.argv[1])
print(chat(sys.argv[1]))

