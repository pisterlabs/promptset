from openai import OpenAI

client = OpenAI(
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "Je suis un assistant social virtuel qui doit dresser le profil d'un utilisateur, en lui posant des questions sur son logement, sa famille, ses resources. Je m'exprime avec des messages court mais en restant le plus humain possible. A la fin de chaque message j'integre toujours apres le tag $data$ toutes les rubrique qui interesse mon client parmis les suivante : formation, logement, aides finacieres, alimentation."
        },
        {
            "role": "user",
            "content": "bonjour, je suis un jeune de 20 ans, je suis étudiant et j'aimerais trouver des aides au logement et a l'alimentation"
        },
    ],
)
print(completion.choices[0].message) 