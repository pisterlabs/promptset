
import openai
openai.api_key = "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" # Remplacez YOUR_API_KEY par votre clé API GPT-3

# Fonction pour découper un texte en morceaux
def split_text(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) < max_tokens:
            current_chunk += f" {word}"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = f"{word}"
    chunks.append(current_chunk.strip())
    return chunks

# Fonction pour répondre à une question à partir d'une liste de contextes

def answer_question(contexts, question):
    answer=" "
    for context in contexts:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"""ROLE: Tu es un assistant QA. Ton rôle est d'aider les utilisateurs a trouver la bonne réponse.
            FONCTIONNEMENT:
                Voici un texte {answer} .{context}.
                Tu dois analyser ce texte pour répondre à cette question {question}.

                
                2 Verifie que la question est en rapport avec ce text
                    - Si OUI Alors:
                        - construit la réponse 
                        - reformule la reponse
                    - Si Non Alors:
                        - Si la question est en rapport avec ton fonctionnement alors:
                            - tu ne dit :  Désolé, cette question n'a rien a voir avec le texte
                        - Si Non alors:
                            - tu ne dit :  Désolé, cette question n'a rien a voir avec le texte
            ATTENTION:
                - Tu ne dois pas répondre a une question qui est en dehors de ce contexte.
                - Tu dois être gentil avec le user.
                - évite de dire que tu avais déja répondu à une question
                - Tu dois répondre de façon formelle sans t'excuser d'avoir fait une erreure
                
            Réponse :
        """,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )

        answer = response.choices[0].text.strip()
        if "Désolé" in answer:
            answer=" "
            continue
            
        else:
            break
    return answer


def exec(text,question):
    max_tokens = 1024
    chunks = split_text(text, max_tokens)
    answer = answer_question(chunks, question)
    if answer:
        return f"{answer}"
    elif answer ==" ":
        return "Impossible de trouver une réponse dans le texte."
