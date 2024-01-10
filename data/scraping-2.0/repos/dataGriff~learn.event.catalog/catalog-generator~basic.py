import openai
openai.api_key = os.getenv("CHATGPT_KEY")
topic = "beer"
words = 500
message = f"Tell me as much as you can about {topic} in less than {words} words and you must include at least one pun."
chat_completion = openai.ChatCompletion.create(model="gpt-4"
                                               , messages=[{"role": "user", "content": message}])
out = chat_completion['choices'][0]['message']['content']
print(out)

   
        
        
        
        

