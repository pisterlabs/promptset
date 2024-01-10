import openai
import os

# ToDO 
# Bygga ett script som låter GPT utvärdera utfallet av CLAI för att se hur träffsäker den är.
# Detta för att slippa mata den med frågor själv och checka mot facit, är för lat för det.

# Initialize OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']
# List of questions and their correct answers
qa_pairs = [
    {
        "question": "Hur lång tid gäller motor och elektronik i extraskydd för elbil och laddhybrid?",
        "correct_answer": "12 år och får ha kört max 17 000 mil"
    },
    {
        "question": "Vid bärningsmomentet i extraskydd för elbil och laddhybrid vart bärgas man vid urladdning av batteri?",
        "correct_answer": "Till närmsta laddstation"
    },
    {
        "question": "Vad är en remitteringsrisk?",
        "correct_answer": "En risk som måste skickas bedömas innan den tecknas"
    },
    {
        "question": "Vem ska kunden kontakta för att ställa av sin bil?",
        "correct_answer": "Transportstyrelsen"
    },
    {
        "question": "Vad är en indikation på att ett fordon kan vara exklusivt?",
        "correct_answer": "Stark motor"
    },
    {
        "question": "Trafikförsäkring ingår i försäkring för lätt släp. Sant eller falskt?",
        "correct_answer": "Falskt"
    },
    {
        "question": "I vilket försäkringsvillkor ingår lätt släp?",
        "correct_answer": "Motorfordonsförsäkring"
    },
    {
        "question": "Sök svaret i villkoret. Var gäller egendom i bil?",
        "correct_answer": "Norden"
    },
    {
        "question": "Sök svaret i villkoret. Hur ska släpet vara låst om det förvaras egendom i det?",
        "correct_answer": "Med lås som är godkänt av SSF"
    },
    {
        "question": "Vilken är den totala maxersättningen för egendom i bil?",
        "correct_answer": "1 prisbasbelopp"
    },
    {
        "question": "Sök svaret i villkoret. Vilken egendom är försäkrad i fordonsförsäkringen? 4 rätta svar",
        "correct_answer": "Fordonet, Normal utrustning som finns i eller på fordonet, Avmonterad fordonsdel eller utrustning som hör till fordonet, Extra uppsättning hjul som hör till fordonet"
    },
    {
        "question": "Vad innebär strikt ansvar?",
        "correct_answer": "Man kan bli ansvarig trots att man inte varit vårdslös"
    },
    {
        "question": "Vad händer om ett påställt fordon inte har trafikförsäkring?",
        "correct_answer": "Ägaren får böter från Trafikförsäkringsföreningen"
    },
    {
        "question": "I vilken lag finns de viktigaste bestämmelserna om ersättning från trafikförsäkring?",
        "correct_answer": "Trafikskadelagen"
    },
    {
        "question": "Vilka skador ersätts från trafikförsäkringen?",
        "correct_answer": "Skador till följd av trafik"
    },
]


def self_evaluate_model():
    for qa_pair in qa_pairs:
        question = qa_pair['question']
        correct_answer = qa_pair['correct_answer']

        # Get the model's answer to the question
        messages = [
            {"role": "system", "content": "You are an insurance expert."},
            {"role": "user", "content": question}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        model_answer = response['choices'][0]['message']['content'].strip()

        # Have the model evaluate its own answer
        messages = [
            {"role": "system", "content": "You are an insurance expert."},
            {"role": "user", "content": f" Is the answer '{model_answer}' correct for the question '{question}'? Reply with a percentage between 0 and 100."},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        self_evaluation = response['choices'][0]['message']['content'].strip()

        print(f"Question: {question}")
        print(f"Model's Answer: {model_answer}")
        print(f"Model's Self-Evaluation: {self_evaluation}")

# Run the self-evaluation
self_evaluate_model()
