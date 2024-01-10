import openai

api_key = "YOUR_API_KEY"


MODEL = "gpt-3.5-turbo"

user_input_thema = input("Das Thema: ")
user_input_fragen = input("Die Anzahl von Fragen: ")
user_input_moeglichkeiten = input("Die Anzahl von Antwortmoeglichkeiten: ")


response = openai.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful quiz generator."},
        {"role": "user", "content": "Erstelle mir ein Quiz über das Thema " + user_input_thema + ", mit " + user_input_fragen + " Fragen. Jede Frage hat " + user_input_moeglichkeiten + " Antwortmoeglichkeiten."},
    ],
    temperature=0,
)

print(response.choices[0].message.content)