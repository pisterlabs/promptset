##just use the web interface lol

# import openai
# import json
# from config import API_KEY  # Import the API key from config.py

# openai.api_key = API_KEY

# def upload_finetune_data(finetune_file_path, fine_tune_model_id):
#     with open(finetune_file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)

#         # Fine-tune data upload
#         response = openai.FineTunes.create(
#             fine_tune_model_id,
#             training_examples=data,
#             name="Fine-tune Data Upload"
#         )

#         print("Fine-tune data uploaded successfully.")
#         print("Fine-tune ID:", response.id)

# # Replace 'output_transformed.json' with the path to your transformed JSON file
# # Replace 'fine_tune_model_id' with your fine-tune model ID
# upload_finetune_data('output_transformed.json', 'fine_tune_model_id')
