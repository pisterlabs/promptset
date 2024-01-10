import openai
openai.api_key=""

def ask_conversation(conversation,model="gpt-3.5-turbo"):
    got_result = False
    num = 0
    while not got_result:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=conversation,
                temperature=0,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            answer = response['choices'][0]['message']
            return answer
        except Exception as e:
            if num >= 3:
                got_result = True
            print(e)
            return False






