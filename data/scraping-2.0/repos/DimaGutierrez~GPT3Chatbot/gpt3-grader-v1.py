import openai
openai.api_key = "YOUR_API_KEY_HERE"

def grade_essay(prompt, response):
    model_engine = "text-davinci-003"
    prompt = (f"Essay prompt: {prompt}\nEssay response: {response}\n"
             "Please grade the essay response and provide feedback.")
    completions = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1,stop=None,temperature=0.5)
    message = completions.choices[0].text
    return message

prompt = "Write an essay on the impact of technology on society."
response = "Technology has had a significant impact on society in recent years. It has transformed the way we communicate, access information, and do business. One of the main benefits of technology is the ability to connect people from all around the world through social media and other platforms. It has also made it easier for people to access education and job opportunities online. However, technology has also had negative impacts on society. It has contributed to a decrease in face-to-face communication and an increase in cyberbullying. In addition, the reliance on technology has led to a decrease in critical thinking skills and an increase in misinformation. Overall, while technology has brought many benefits to society, it is important to use it responsibly and consider the potential downsides."

print(grade_essay(prompt, response))
