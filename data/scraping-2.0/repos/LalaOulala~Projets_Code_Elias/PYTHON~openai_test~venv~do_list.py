# NOMBRE DE TOKENS MAX : 4097

import openai
openai.api_key = "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"

# suppression des retours à la ligne que rajoute l'IA avant sa réponse
def remove_return(string):
  return string.replace("\n", "")

def titre(string):
  # découper la chaîne en lignes en utilisant le caractère de nouvelle ligne "\n" comme séparateur
  texte = string.split("\n")
  print("affichage 1",texte)
  # initialise la variable
  texte_f =[]
  # parcourir chaque ligne
  for e in range(len(texte)-1):
    # rechercher la première occurence de la chaîne "Fiche de révision"
    if "Fiche de révision" in texte[e]:
      # remplir la liste vide avec les lignes suivantes jusqu'à la fin du texte
      for s in range(len(texte)-e):
        texte_f.append(texte[e+s])
        print("affichage 2",texte)
  print("valeur de texte_f",texte_f)
  # retourner la chaine de caractère à partir du 20ème caractère de la première ligne de "texte_f" c'est à dire le titre
  try:
    return texte_f[0][20:]
  except IndexError:
    print("L'IA n'a pas généré de réponse à votre prompt")

# prise de notes sur un texte
def AI_notes(string, exemple):
  # création de la completion
  user = """Fait une fiche de révision en suivant scrupuleusement cet exemple : {} avec ce texte {}""".format(exemple, string)

  completion = openai.Completion.create(
    model="text-davinci-003",
    prompt=user,
    temperature=0.9,
    max_tokens=1500,
    top_p=1.0,
    frequency_penalty=0.8,
    presence_penalty=0.0
  )

  # affichage de la completion
  ai_text = completion.choices[0].text

  remove_return(ai_text)

  print("texte généré par l'IA : ",ai_text)

  file_name = str(titre(ai_text)) + ".txt"

  # création du fichier tecte contenant la fiche de révision
  with open(file_name, "a") as resume:
      resume.write(ai_text)


# list engines
engines = openai.Engine.list()

# print the first engine's id
print(engines.data[0].id)

# Lecture du texte a fiché dans le fichier texte
with open("texte.txt", "r") as data:
  user = data.read()
print(user)

# Lecture de l'exemple
with open("exemple.txt", "r") as Ex:
  exemple = Ex.read()
print(exemple)


AI_notes(user, exemple)