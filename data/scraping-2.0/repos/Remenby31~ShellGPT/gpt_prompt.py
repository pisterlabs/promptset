from openai import OpenAI
import os
from get_environnement import get_environnement

key = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=key)

def get_gpt_command(task_description, historique):
    """Renvoie la commande à exécuter pour accomplir la tâche donnée."""

    distribution, terminal_name, current_path, ls_output = get_environnement()
    gpt_prompt = f"Je suis sur un système {distribution} avec {terminal_name}. Mon chemin actuel est {current_path}. Voici la liste des fichiers dans mon répetoire : {ls_output} Pour accomplir la tâche suivante : '{task_description}', quelle commande spécifique dois-je taper dans le terminal ? Donne uniquement la prochaine commande à taper."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Vous êtes une AI qui renvoie uniquement des commandes pour accomplir des tâches sur un terminal."},
                {"role": "user", "content": gpt_prompt},
                {"role": "system", "content": "Voici l'historique de ce qu'il s'est passé avant: " + historique},
                {"role": "system", "content": "Il suffit de taper la commande suivante dans le terminal : "}
            ],
            max_tokens=40,
        )
        # Récupérer la dernière réponse de l'assistant
        command = response.choices[-1].message.content
        historique += '[assistant]' + command + '\n'
        return historique, command
    except Exception as e:
        print(f"Erreur lors de la communication avec OpenAI: {e}")
        return historique, None

def check_gpt_end(historique, task_description):
    """Vérifie si la tâche est terminée."""
    
    gpt_prompt = f"Voici l'historique : '{historique}'. La tâche \"'{task_description}' \" nécessite-t-elle encore d'autres actions ? Répondre par 'oui' ou 'non'."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Vous êtes une AI vérifie si les tâches demandés sont accomplies."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=1,
        )

        # Récupérer la dernière réponse de l'assistant
        reponse = response.choices[-1].message.content
        return reponse[0] == "n" or reponse[0] == "N"
    except Exception as e:
        print(f"Erreur lors de la communication avec OpenAI: {e}")
        return None