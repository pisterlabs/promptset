import openai
from decouple import config
from utils import now

openai.api_key = config("OPENAI_API_KEY")

NOW_PHRASE = f"The point of questioning is {now}."

CONTEXT_FOR_GPT = """
Du bist ein Servicebot für Buchungsanfragen.
Extrahiere das Datum, die Uhrzeit und den Namen aus der Anfrage und gib die Informationen im folgenden JSON-Format aus:
{"name": "NAME", "time": "HH:MM:SS”, "date":"YYYY-MM-DD"}.
Von dem Zeitpunkt der Anfrage sollen die Zeiten berechnet werden. wie zum beispiel "Übermorgen", "Morgen", etc.
Überprüfe ob in der Anfrage sowohl ein Datum als auch eine Uhrzeit vorhanden ist. Wenn nicht dann frage jeweils nach dem fehlenden Teil.
Nimm nicht irgendwelche Werte an.
Wenn der Kunde nach "Morgen" fragt, evaluiere das Datum ausgehend von heute.
Sollte kein Name in der Buchungsanfrage vorhanden sein, verwende None.
Sollte kein Datum in der Buchungsanfrage vorhanden sein, verwende None. Datum ist hier als "date zu verstehen" und damit ist nur der kalender tag gemeint.
Sollte keine Uhrzeit in der Buchungsanfrage vorhanden sein, verwende None.
Achte darauf, dass die Ausgabe ausschließlich dieses JSON-Format enthält und verzichte auf jegliche Hinweise und Floskeln.
Jede deiner Antworten darf nur im Schema: "name": "XXX", "time": "HH:MM:SS”, "date":"YYYY-MM-DD sein.
"""
# CONTEXT_FOR_GPT = """
# This is the context. Do not repeat this or summarize it to the user.
# You are a service bot that handles booking requests. You guide the customer to the point where he or she provides 3 pieces of information.
# 1 Name, 2 Date, 3 Time.
# Make sure that the customers gives these 3 pieces of information and if the conversion tends to go somewhere else bring it back to the booking.
# If you have all the booking information, prompt the customer: "Lassen Sie mich überprüfen ob der Termin verfügbar ist. Datum: {date} Zeit: {time}."
# This prompt is always necessary, because I use it to query my database. Make sure that the format of date is YYYY-MM-DD and the format of time is HH:MM:SS.
# If the the schedule is available, ask the customer to confirm the booking, with the prompt 'Datum und Uhrzeit sind verfügbar. Wollen Sie den Termin buchen?'.
# If the customer confirms the booking, send the prompt "Ich habe den Termin gebucht. Vielen Dank für Ihre Buchung."
# """

def get_gpt_response(question, test):
    messages = [{"role": "user", "content": question}]
    messages.append({"role": "system", "content": NOW_PHRASE})
    messages.append({"role": "system", "content": CONTEXT_FOR_GPT})

    if not test:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.5
        )

        chatgpt_response = response.choices[0].message.content
    else:
        chatgpt_response = question

    return chatgpt_response