import openai
import time

# Configura tu clave API de OpenAI aquí
openai.api_key = "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"

# Sube el archivo a OpenAI y obtén el ID del archivo
response = openai.File.create(
  file=open("Chat_gpt_/resoluciones.jsonl"),
  purpose="fine-tune"
)
file_id = response['id']
print("ID del archivo:", file_id)

# Configura los parámetros para el fine-tuning
model = "text-davinci-003"  # Reemplaza con el modelo base que deseas usar
n_epochs = 4  # Número de épocas para el entrenamiento

# Inicia el entrenamiento
training_response = openai.FineTune.create(
  training_file=file_id,
  model=model,
  n_epochs=n_epochs
)
fine_tune_id = training_response['id']
print("ID del entrenamiento:", fine_tune_id)

# Comprueba el estado del entrenamiento
while True:
    status_response = openai.FineTune.retrieve(id=fine_tune_id)
    status = status_response['status']
    print("Estado del entrenamiento:", status)
    
    if status in ["succeeded", "failed", "cancelled"]:
        break
    
    time.sleep(60)  # Espera 1 minuto antes de comprobar de nuevo

print("Entrenamiento completado.")