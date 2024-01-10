from openai import OpenAI

# Initialize OpenAI client with your API key
client = OpenAI(api_key="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")

def generate_teacher_content(prompt, max_tokens=1000, temperature=0.8, top_p=1):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].text

def simulate_student_response(teacher_content, student_prompt="", max_tokens=1000, temperature=0.8, top_p=1):
    messages = [{"role": "system", "content": teacher_content}]
    if student_prompt:
        messages.append({"role": "user", "content": student_prompt})
        
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message.content
