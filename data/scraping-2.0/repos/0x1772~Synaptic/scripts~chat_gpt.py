import openai
import sys

openai.api_key = 'XXX'  # OpenAI API anahtarınızı buraya girin

def ask_gpt(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
        log_level="info"
    )
    return response.choices[0].text.strip()

def main():
    question = " ".join(sys.argv[1:])
    answer = ask_gpt(question)
    print(answer)

if __name__ == "__main__":
    main()