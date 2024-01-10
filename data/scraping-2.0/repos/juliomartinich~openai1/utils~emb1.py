import openai, os, csv

openai.api_key = os.environ.get('OPENAIAPIKEY')

palabras = ["gato", "perro", "burro", "cisne", "zorro", "liebre", "caballo", "nutria", "coipo" ]

vectores = {}
for palabra in palabras:
  vectores[palabra] = openai.Embedding.create(
    input=palabra, model="text-embedding-ada-002"
  )["data"][0]["embedding"]

file_name = "vectores_de_animales.csv"

# Open the file in write mode
with open(file_name, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the vectors to the file
    for palabra in palabras:
        writer.writerow(vectores[palabra])

# me falta guardar la palabra, tengo solo el vector

