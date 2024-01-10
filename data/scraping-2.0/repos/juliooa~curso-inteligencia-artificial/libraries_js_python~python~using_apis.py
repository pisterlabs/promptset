import openai

openai.api_key = "TU_API_KEY"
completion = openai.completions.create(
    model="text-davinci-003",
    prompt="Saluda a los estudiantes del curso de IA",
    max_tokens=100,
    temperature=1,
)

print(completion)
